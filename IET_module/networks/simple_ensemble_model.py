import numpy as np
from scipy.stats import norm
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


class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int],
                 ) -> None:
        super(BaseMLP, self).__init__()
        layers = []
        layers.append(SlimFC(in_size=input_dim,
                           out_size=hidden_dims[0],
                           activation_fn='tanh',
                           initializer=torch.nn.init.xavier_uniform_)
                      )
        for i in range(len(hidden_dims) - 1):
            layer = SlimFC(in_size=hidden_dims[i],
                           out_size=hidden_dims[i + 1],
                           activation_fn='tanh',
                           initializer=torch.nn.init.xavier_uniform_)
            layers.append(layer)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self._net = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._net(input)

class SimpleEnsembleNet(nn.Module):
    def __init__(self, observation_space, action_space,
                 model_config, network_type: str='policy',
                 **kwargs):
        nn.Module.__init__(self)
        self._network_type = network_type
        assert network_type in ['policy', 'value'], "must be either policy or value network"
        if self._network_type == 'policy':
            self._output_shape = action_space.n
        else:
            self._output_shape = 1

        self._config = model_config['custom_model_config']
        self._ensemble_size = self._config['ensemble_size']
        self._use_dict_obs_space = self._config['use_dict_obs_space']
        # threshold to determine input goes to which ensemble, first -inf, last inf
        self._gaussian_cdf_thres = [norm.ppf(i / self._ensemble_size) for i in range(self._ensemble_size + 1)]

        assert self._use_dict_obs_space
            # the obs_space_dict was flattened by rllib, but the observation is the original
        base_input_shape = observation_space.original_space.spaces['original_obs'].shape
        for i in range(self._ensemble_size):
            setattr(self, f'head_{i}', BaseMLP(base_input_shape[0], self._output_shape, [256, 256]))

    def forward(self, input_dict, state, seq_lens):
        if self._use_dict_obs_space:
            x = input_dict['obs']['original_obs']
            gaussian_latent_parameter = input_dict['obs']['random_noise'][:, 0]
        else:
            x = input_dict['obs'][:, :-self._em_input_shape]
            gaussian_latent_parameter = input_dict['obs'][:, -self._em_input_shape:][:, 0]

        batch_size = x.shape[0]
        out = torch.zeros(batch_size, self._output_shape)
        total_sample = 0
        for i in range(self._ensemble_size):
            index = torch.nonzero((gaussian_latent_parameter > self._gaussian_cdf_thres[i]) *
                                  (gaussian_latent_parameter <= self._gaussian_cdf_thres[i + 1]),
                                    as_tuple=True)[0]
            total_sample += index.shape[0]
            if index.shape[0] == 0:
                continue
            selected_input = torch.index_select(x, dim=0, index=index)
            model = getattr(self, f'head_{i}')
            out.index_add_(0, index, model(selected_input))

        if not total_sample == batch_size:
            raise Exception("total number of sample must be wrong")
        if self._network_type == 'value':
            out = out.view(out.shape[0])
        return out, state


class SimpleEnsembleActorCriticNet(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs,
                 model_config, name, **kwargs):
        TorchModelV2.__init__(self, observation_space, action_space,
                              num_outputs, model_config, name)
        nn.Module.__init__(self)
        self._current_value = None
        self._policy_net = SimpleEnsembleNet(observation_space,
                                                     action_space,
                                                     model_config,
                                                     network_type='policy')
        self._value_net = SimpleEnsembleNet(observation_space,
                                                    action_space,
                                                    model_config,
                                                    network_type='value')

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        logits, *_ = self._policy_net(input_dict, state, seq_lens)
        self._current_value, *_ = self._value_net(input_dict, state, seq_lens)
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