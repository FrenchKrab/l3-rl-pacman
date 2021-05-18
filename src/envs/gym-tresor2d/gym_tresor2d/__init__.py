from gym.envs.registration import register

register(
    id='tresor2d-v0',
    entry_point='gym_tresor2d.envs:Tresor2dEnv',
    kwargs={'maze_size' : [8,8]}
)