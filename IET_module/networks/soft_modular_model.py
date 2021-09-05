import numpy as np
from typing import Tuple, List
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import SlimFC
import torch.nn.functional as F
from IET_module.networks.utils import basic_init, uniform_init, null_activation
from IET_module.utils.check_nan_inf import assert_nan_inf


torch, nn = try_import_torch()


class EmbeddingShapingNet(nn.Module):
    # This network takes the gaussian latent parameter as input,
    # and shape it through a FC network with a softmax layer at the end
    # so that the output is a soft version of a task one-hot embedding
    # we use the same shape for input, intermediate layer and output
    def __init__(self, eb_dim: int,
                 hidden_dims: List[int]=None,
                 softmax: bool=True,
                 ) -> None:
        super(EmbeddingShapingNet, self).__init__()
        layers = []
        if not hidden_dims:
            hidden_dims = [eb_dim for j in range(2)]
        layers.append(SlimFC(in_size=eb_dim,
                           out_size=hidden_dims[0],
                           activation_fn='relu',
                           initializer=torch.nn.init.xavier_uniform_)
                      )
        for i in range(len(hidden_dims) - 1):
            layer = SlimFC(in_size=hidden_dims[i],
                           out_size=hidden_dims[i + 1],
                           activation_fn='relu',
                           initializer=torch.nn.init.xavier_uniform_)
            layers.append(layer)
        layers.append(nn.Linear(hidden_dims[-1], eb_dim))
        if softmax:
            layers.append(nn.Softmax(dim=-1))
        self._net = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self._net(embedding)

class ModularGatedCascadeCondNet(nn.Module):
    def __init__(self, observation_space, action_space, num_outputs,
                 model_config, name, network_type: str='policy',
                 **kwargs):
        nn.Module.__init__(self)
        self._network_type = network_type

        assert network_type in ['policy', 'value'], "must be either policy or value network"
        if self._network_type == 'policy':
            output_shape = action_space.n
        else:
            output_shape = 1

        self._config = model_config['custom_model_config']
        self._use_latent_embedding = self._config['use_latent_embedding']
        self._use_dict_obs_space = self._config['use_dict_obs_space']
        base_type = self._config['base_type']
        em_input_shape = self._config['em_input_shape']
        self._em_input_shape = em_input_shape
        self._emb_shaping_net_hidden_shapes = self._config['emb_shaping_net_hidden_shapes']
        self._emb_shaping_net_last_softmax = self._config['emb_shaping_net_last_softmax']
        em_hidden_shapes = self._config['em_hidden_shapes']
        hidden_shapes = self._config['hidden_shapes']
        num_layers, num_modules = self._config['num_layers'], self._config['num_modules']
        module_hidden = self._config['module_hidden']
        gating_hidden, num_gating_layers = self._config['gating_hidden'], self._config['num_gating_layers']
        # gated_hidden
        add_bn = self._config.get('add_bn', True)
        pre_softmax = self._config.get('pre_softmax', False)
        cond_ob = self._config.get('cond_ob', True)
        module_hidden_init_func = self._config.get('module_hidden_init_func', basic_init)
        last_init_func = self._config.get('last_init_func', uniform_init)
        activation_func = self._config.get('activation_func', F.relu)

        if self._use_dict_obs_space:
            # the obs_space_dict was flattened by rllib, but the observation is the original
            base_input_shape = observation_space.original_space.spaces['original_obs'].shape
        else:
            input_shape = observation_space.shape
            def _get_base_input_shape(raw_input_shape: Tuple, embedding_dim: int) -> Tuple:
                if not len(raw_input_shape) == 1:
                    raise AssertionError("only support flat observation")
                return (raw_input_shape[0] - embedding_dim,)
            base_input_shape = _get_base_input_shape(input_shape, em_input_shape)

        self.base = base_type(
                        last_activation_func = null_activation,
                        input_shape = base_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = hidden_shapes,)
        self.em_base = base_type(
                        last_activation_func = null_activation,
                        input_shape = em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes,)

        if self._use_latent_embedding:
            self._embedding_shaping_net = EmbeddingShapingNet(eb_dim=self._em_input_shape,
                                                              hidden_dims=self._emb_shaping_net_hidden_shapes,
                                                              softmax=self._emb_shaping_net_last_softmax,
                                                              )

        self.activation_func = activation_func

        module_input_shape = self.base.output_shape
        self.layer_modules = []

        self.num_layers = num_layers
        self.num_modules = num_modules

        for i in range(num_layers):
            layer_module = []
            for j in range( num_modules ):
                fc = nn.Linear(module_input_shape, module_hidden)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    )
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i,j), module)

            module_input_shape = module_hidden
            self.layer_modules.append(layer_module)
        self.last = nn.Linear(module_input_shape, output_shape)
        last_init_func( self.last )

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated"
        gating_input_shape = self.em_base.output_shape
        self.gating_fcs = []
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden)
            module_hidden_init_func(gating_fc)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape,
                    num_modules * num_modules )
        last_init_func( self.gating_weight_fc_0)
        # self.gating_weight_fcs.append(self.gating_weight_fc_0)

        for layer_idx in range(num_layers-2):
            gating_weight_cond_fc = nn.Linear((layer_idx+1) * \
                                               num_modules * num_modules,
                                              gating_input_shape)
            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)

            gating_weight_fc = nn.Linear(gating_input_shape,
                                         num_modules * num_modules)
            last_init_func(gating_weight_fc)
            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)

        self.gating_weight_cond_last = nn.Linear((num_layers-1) * \
                                                 num_modules * num_modules,
                                                 gating_input_shape)
        module_hidden_init_func(self.gating_weight_cond_last)

        self.gating_weight_last = nn.Linear(gating_input_shape, num_modules)
        last_init_func( self.gating_weight_last )

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

    @property
    def embedding_shaping_net(self) -> nn.Module:
        return self._embedding_shaping_net

    def forward(self, input_dict, state, seq_lens, return_weights = False):
        # out = self.first_layer(input_dict["obs"])
        # self._output = self._global_shared_layer(out)
        # model_out = self.last_layer(self._output)
        # return model_out, []
        # extract the head of the observation vector as raw obs
        # and the tail of the observation vector as the latent Gaussian parameter
        if self._use_dict_obs_space:
            x = input_dict['obs']['original_obs']
            gaussian_latent_parameter = input_dict['obs']['random_noise']
        else:
            x = input_dict['obs'][:, :-self._em_input_shape]
            gaussian_latent_parameter = input_dict['obs'][:, -self._em_input_shape:]

        if self._use_latent_embedding:
            embedding_input = self._embedding_shaping_net(gaussian_latent_parameter)
        else:
            batch_size = x.shape[0]
            embedding_input = torch.ones(batch_size, self._em_input_shape) / self._em_input_shape

        out = self.base(x)
        embedding = self.em_base(embedding_input)

        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)

        if len(self.gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []

        raw_weight = self.gating_weight_fc_0(self.activation_func(embedding))

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)
        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim = -1)

        module_outputs = [(layer_module(out)).unsqueeze(-2) \
                for layer_module in self.layer_modules[0]]

        module_outputs = torch.cat(module_outputs, dim = -2 )

        # [TODO] Optimize using 1 * 1 convolution.

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.layer_modules[i + 1]):
                module_input = (module_outputs * \
                    weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                        layer_module(module_input)
                ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim = -2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        if self._network_type == 'policy':
            out = self.last(out)
        else:
            out = self.last(out).view(out.shape[0])
        if return_weights:
            return out, state, weights, last_weight
        return out, state


class SoftModularActorCriticNet(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs,
                 model_config, name, **kwargs):
        TorchModelV2.__init__(self, observation_space, action_space,
                              num_outputs, model_config, name)
        nn.Module.__init__(self)
        self._current_value = None
        self._policy_net = ModularGatedCascadeCondNet(observation_space, 
                                                     action_space,
                                                     num_outputs,
                                                     model_config,
                                                     name,
                                                     network_type='policy')
        self._value_net = ModularGatedCascadeCondNet(observation_space, 
                                                    action_space, 
                                                    num_outputs,
                                                    model_config, 
                                                    name,
                                                    network_type='value')

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens, return_weights = False):
        logits, *_ = self._policy_net(input_dict, state, seq_lens, return_weights=return_weights)
        self._current_value, *_ = self._value_net(input_dict, state, seq_lens, 
                                                    return_weights=return_weights)
        if (self._current_value > 1000000).any():
            print(f'soft_modular_model line 323: vf too large: {self._current_value}')
        assert_nan_inf([logits, self._current_value])
        # clip the logits to prevent inf from kl
        logits = torch.clip(logits, min=-15.0, max=15.0)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> torch.Tensor:
        if self._current_value is None:
            raise AssertionError("must call self._value_net.forward() first")
        return self._current_value

    @property
    def policy_net(self) -> nn.Module:
        return self._policy_net

    @property
    def value_net(self) -> nn.Module:
        return self._value_net