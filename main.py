import time
import torch
import ray
from ray.tune.logger import pretty_print
from IET_module.utils.rendering import render
from IET_module.config import Args, get_config
from IET_module.custom_logger_creator import get_default_logger_creator
from IET_module.cmd_args import cmd_args

torch.autograd.set_detect_anomaly(True)
for i in range(10):
    print("----------Remember to set `shape = True` for tag and world_comm----------")
time.sleep(1)

if __name__ == "__main__":
    num_cpus = cmd_args.num_cpus
    args = Args({'num_cpus': num_cpus,
                 'train_setting': cmd_args.training_setting,
                 'game': cmd_args.env,
                 'use_latent_embedding': True,
                 "soft_modular_net_hidden_dim": 64,
                 "atari_obs_type": 'ram', # 'ram' or 'rgb_image' or 'grayscale_image'
                 'max_steps': {'mpe': 100, 'atari': 500, 'classic': 100},
                 'latent_para_dim': 10,
                 'emb_shaping_net_hidden_shapes': [64, 64],
                 'soft_modular_net_num_layers': 2,
                 'soft_modular_net_num_modules': 2,
                 'emb_shaping_net_last_softmax': True,
                 })
    torch.set_num_threads(args.num_cpus)
    trainer_class, dummy_env, main_config, test_config = get_config(args)
    ray.init(num_cpus=args.num_cpus)
    main_log_creator = get_default_logger_creator(log_dir=cmd_args.logdir,
                                                  env_id=args.game_name,
                                                  train_setting=f'{cmd_args.training_setting}_train',
                                                  )
    test_log_creator = get_default_logger_creator(log_dir=cmd_args.logdir,
                                                  env_id=args.game_name,
                                                  train_setting=f'{cmd_args.training_setting}_test',
                                                  )
    main_trainer = trainer_class(env=args.game_name, config=main_config, logger_creator=main_log_creator)
    test_trainer = trainer_class(env=args.game_name, config=test_config, logger_creator=test_log_creator)
    if cmd_args and cmd_args.main_ckpt:
        main_trainer.restore(cmd_args.main_ckpt)
        print("---successfully restored from main checkpoint---")
    if cmd_args and cmd_args.test_ckpt:
        test_trainer.restore(cmd_args.test_ckpt)
        print("---successfully restored from test checkpoint---")
    # 7. Train once
    if cmd_args.training_setting == 'simple_ensemble':
        PERIODS = [30, 30]
    else:
        PERIODS = [10, 10]

    SAVE_PERIOD = 6000 / 20 * sum(PERIODS)
    PRINT_KEYS = ['policy_reward_mean', 'policy_reward_max', 'policy_reward_min']
    main_trainer.train()
    test_trainer.train()
    current_training = 'test'
    current_sub_step = 0

    total_steps = cmd_args.train_step_multiplier * sum(PERIODS)
    start = time.time()

    for i in range(total_steps):
        if current_training == 'main':
            result = main_trainer.train()
            current_sub_step += 1
            current_trainer = main_trainer
            if current_sub_step >= PERIODS[0]: # switch
                current_sub_step = 0
                current_training = 'test'
            if i % 1 == 0:
                print(f"Step: {i}; train both agents:")
                print(f"{pretty_print({key: result[key] for key in PRINT_KEYS})}") #
        else:
            assert current_training == 'test'
            if current_sub_step == 0:
                print("Syncing policies into the test_trainer")
                # only set local worker weights
                test_trainer.set_weights(
                                        main_trainer.get_weights(
                                        main_trainer.config["multiagent"]["policies_to_train"]))
                # broadcast weight to all the remote workers
                test_trainer.workers.sync_weights()
            result = test_trainer.train()
            current_sub_step += 1
            current_trainer = test_trainer
            if current_sub_step >= PERIODS[1]: # switch
                current_sub_step = 0
                current_training = 'main'
            if i % 1 == 0:
                print(f"Step: {i}; train test adversary:")
                print(f"{pretty_print({key: result[key] for key in PRINT_KEYS})}")
        if i % 5 == 0 and not args.game_name.startswith('rps'):
            if cmd_args and not cmd_args.no_render:
                render(current_trainer, dummy_env)
        if i % 10 == 0:
            print(f'total training time up to now: {time.time() - start}')
        if i % SAVE_PERIOD == SAVE_PERIOD - 1:
            main_trainer.save()
            test_trainer.save()