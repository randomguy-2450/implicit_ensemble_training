#!/usr/bin/env python3


def render(trainer, dummy_env, max_steps=None) -> None:
    config = trainer.config
    if not max_steps:
        max_steps = config["horizon"]
    done = False
    obs = dummy_env.reset()
    step = 0
    while not done:
        step += 1
        actions = {}
        for agent_id, agent_obs in obs.items():
            policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
            actions[agent_id] = trainer.compute_action(agent_obs, policy_id=policy_id)
        obs, reward, done, info = dummy_env.step(actions)
        dummy_env.render()
        done = done['__all__']
        if step >= max_steps:
            break
    return