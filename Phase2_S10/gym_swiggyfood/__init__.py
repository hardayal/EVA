from gym.envs.registration import register

register(
    id='SwiggyFood-v0',
    entry_point='gym_swiggyfood.envs:SwiggyFoodEnv',
    max_episode_steps=3000
)
