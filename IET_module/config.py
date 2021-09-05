from copy import deepcopy
import random
from typing import Dict, Any, Tuple

from numpy import float32

from ray.rllib.models import ModelCatalog
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.tune.registry import register_env

from supersuit import (dtype_v0,
                       pad_observations_v0,
                       pad_action_space_v0,
                       frame_skip_v0,
                       frame_stack_v1,
                       )
from ray.rllib.env import PettingZooEnv
from IET_module.networks.soft_modular_model import SoftModularActorCriticNet
from IET_module.networks.simple_ensemble_model import SimpleEnsembleActorCriticNet
from IET_module.env_wrapper.latent_parameter_augmented_envwrapper \
    import LatentGaussianAugmentedEnvWrapper
from IET_module.env_wrapper.flatten_envwrapper import FlattenEnvWrapper
from IET_module.env_wrapper.onehot_wrapper import one_hot_obs_wrapper
from IET_module.networks.base_networks import MLPBase

class Args(object):
    alg_name = "PPO"
    latent_para_dim = 5
    # if use_latent_embedding == False, latent_gaussian_noise will not be used
    use_latent_embedding = True
    use_dict_obs_space = True
    game = None
    num_cpus = None
    max_steps = {'mpe': 200, 'atari': 500, 'classic': 100}
    train_setting = 'single_policy'
    soft_modular_net_hidden_dim = 64
    emb_shaping_net_hidden_shapes = [64, 64]
    emb_shaping_net_last_softmax = True
    soft_modular_net_num_layers = 2
    soft_modular_net_num_modules = 2
    atari_obs_type = 'ram'
    atari_frame_skip_num = 2
    atari_frame_stack_num = 2

    def __init__(self, override_dict: Dict=None) -> None:
        if override_dict:
            for key, value in override_dict.items():
                if not hasattr(self, key):
                    raise KeyError("unknown argument")
                setattr(self, key, value)
        self._post_init_processing()
        return

    def _post_init_processing(self) -> None:
        self.game_name = self.game.__name__.split('.')[-1]
        self.game_type = self.game.__name__.split('.')[-2]
        if not self.use_latent_embedding and self.train_setting == 'implicit_ensemble':
            for i in range(10):
                print('----------------------------------------')
                print('Warning: config line 60, gaussian latent noise not used !!!!!')
                print('----------------------------------------')

def is_adversary(agent_id: str) -> bool:
    return (('adversary' in agent_id)
            or agent_id.startswith('eve')
            or agent_id.startswith('player_1')
            or ('second' in agent_id))

def get_config(args: Args):
    # num_rollouts = 2
    ModelCatalog.register_custom_model("SoftModularActorCriticNet", SoftModularActorCriticNet)
    ModelCatalog.register_custom_model("SimpleEnsembleActorCriticNet", SimpleEnsembleActorCriticNet)
    # 1. Gets default training configuration and specifies the POMgame to load.
    config = deepcopy(get_agent_class(args.alg_name)._default_config)

    # 2. Set environment config. This will be passed to
    # the env_creator function via the register env lambda below.
    # local_ratio specify hthe ratio between global reward and the local reward
    # config["env_config"] = {"local_ratio": 0.5}
    def env_creator():
        if args.game.__package__.endswith('atari'):
            if (args.game_name.startswith('foozpong') or
                args.game_name.startswith('basketball_pong') or
                args.game_name.startswith('volleyball_pong')
                ):
                env = args.game.env(obs_type=args.atari_obs_type,
                                    max_cycles=args.max_steps['atari'],
                                    full_action_space=False,
                                    num_players=2)
            else:
                env = args.game.env(obs_type=args.atari_obs_type,
                                    full_action_space=False,
                                    max_cycles=args.max_steps['atari'])
            env = frame_skip_v0(env, args.atari_frame_skip_num)
            env = frame_stack_v1(env, args.atari_frame_stack_num)

        else:
            env = args.game.env()
        if args.game_name.startswith('rps'):
            env = one_hot_obs_wrapper(env)
        env = dtype_v0(env, dtype=float32)
        env = pad_observations_v0(env)
        env = pad_action_space_v0(env)
        if args.game_name.startswith('connect_four') or args.game_name.startswith('tictactoe'):
            env = FlattenEnvWrapper(env)
        GAUSSIAN_STD = 1.0
        assert abs(GAUSSIAN_STD - 1.0) < 1e-5, "must be 1.0, otherwise simple ensemble implementation is wrong"
        env = LatentGaussianAugmentedEnvWrapper(env,
                                                latent_parameter_dim=args.latent_para_dim,
                                                gaussian_std=1.0,
                                                use_dict_obs_space=args.use_dict_obs_space)
        return env

    # 3. Register env, and get trainer_class
    register_env(args.game_name,
                 lambda config: PettingZooEnv(env_creator()))
    trainer_class = get_agent_class(args.alg_name)

    # 4. Extract space dimensions
    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    agents_id = test_env.agents
    print(f"obs_space: {obs_space}; act_space: {act_space}")

    # 5. Configuration for multiagent setup:
    config["framework"] = "torch"
    config["num_gpus"] = 0
    config["log_level"] = "INFO"
    config["num_workers"] = args.num_cpus // 2
    config["num_cpus_per_worker"] = 1
    config['num_envs_per_worker'] = 5
    # Fragment length, collected at once from each worker and for each agent!
    config["rollout_fragment_length"] = 100
    # Training batch size -> Fragments are concatenated up to this point.
    config["train_batch_size"] = 2000
    config["sgd_minibatch_size"] = 256
    config["entropy_coeff"] = 0.01
    config["lambda"] = 0.9
    config["vf_clip_param"] = 50
    config["num_sgd_iter"] = 10
    # After n steps, force reset simulation
    config["horizon"] = args.max_steps[args.game_type]
    # Default: False
    config["no_done_at_end"] = False
    # Info: If False, each agents trajectory is expected to have
    # maximum one done=True in the last step of the trajectory.
    # If no_done_at_end = True, environment is not resetted
    # when dones[__all__]= True.
    config['ignore_worker_failures'] = True

    def get_main_and_test_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any],
                                                                Dict[str, Any]]:

        main_policies = {}
        for i, agent_id in enumerate(agents_id):
            for j in range(1):
                main_policies[f'{agent_id}_{j}'] = (PPOTorchPolicy,
                                                    obs_space,
                                                    act_space,
                                                    {"framework": "torch"})
        test_policies = {
                'test_' + agent_id: (PPOTorchPolicy, obs_space, act_space, {"framework": "torch"})
                for agent_id in agents_id if is_adversary(agent_id)
                        }
        policies = {**main_policies, **test_policies}

        main_config, test_config = deepcopy(config), deepcopy(config)

        main_config["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: f'{agent_id}_{0}',
            "policies_to_train": list(main_policies.keys())
        }

        def test_config_policy_mapping(agent_id: str) -> str:
            if is_adversary(agent_id):
                return 'test_' + agent_id
            return f'{agent_id}_{0}'

        test_config["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": test_config_policy_mapping,
            "policies_to_train": list(test_policies.keys())
        }
        return main_config, test_config

    def get_simple_ensemble_training_config(config: Dict[str, Any], ensemble_size: int=3) -> Tuple[Dict[str, Any],
                                                                             Dict[str, Any]]:
        if ensemble_size > 1:
            config["model"] = {
                    "custom_model": "SimpleEnsembleActorCriticNet",
                    "custom_model_config": {
                                            "use_dict_obs_space": args.use_dict_obs_space,
                                            'ensemble_size': ensemble_size
                                            }
                            }
        main_config, test_config = get_main_and_test_config(config)
        return main_config, test_config

    def get_implicit_ensemble_training_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any],
                                                                               Dict[str, Any]]:
        config["model"] = {
                "custom_model": "SoftModularActorCriticNet",
                "custom_model_config": {
                                        "use_latent_embedding": args.use_latent_embedding,
                                        "use_dict_obs_space": args.use_dict_obs_space,
                                        "base_type": MLPBase,
                                        "em_input_shape": args.latent_para_dim,
                                        "emb_shaping_net_hidden_shapes": args.emb_shaping_net_hidden_shapes,
                                        'emb_shaping_net_last_softmax': args.emb_shaping_net_last_softmax,
                                        'em_hidden_shapes': [args.soft_modular_net_hidden_dim,
                                                             args.soft_modular_net_hidden_dim], #[400],
                                        'hidden_shapes': [args.soft_modular_net_hidden_dim,
                                                          args.soft_modular_net_hidden_dim], #[400, 400],
                                        'num_layers': args.soft_modular_net_num_layers, #4,
                                        'num_modules': args.soft_modular_net_num_modules, #4,
                                        'module_hidden': args.soft_modular_net_hidden_dim, #128,
                                        'gating_hidden': args.soft_modular_net_hidden_dim, #256,
                                        'num_gating_layers': 2,  #with 1 gating layer, 500 step works for simple_spread
                                        'add_bn': False,
                                        }
                        }
        main_config, test_config = get_main_and_test_config(config)
        return main_config, test_config

    if args.train_setting == 'single_policy':
        main_config, test_config = get_simple_ensemble_training_config(config, ensemble_size=1)
    elif args.train_setting == 'simple_ensemble':
        main_config, test_config = get_simple_ensemble_training_config(config, ensemble_size=3)
    else:
        assert args.train_setting == 'implicit_ensemble'
        main_config, test_config = get_implicit_ensemble_training_config(config)

    return trainer_class, test_env, main_config, test_config