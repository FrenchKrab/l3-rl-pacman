import gym
import gym_tresor2d
from agents.dqn_agent_v2 import DQNAgent
from agents.qlearning_agent import QLearningAgent
import numpy as np

def mspacman_init_data():
    data = {}
    data["lives"] = 3
    return data

def ms_pacman_step_post_process(new_state, reward, done, infos, data):
    dead = infos['ale.lives']<data["lives"]     #Test if pacman is dead
    data["lives"] = infos['ale.lives']          #Update lives counter

    if dead:
        reward = -100
    
    return new_state, reward, done, infos

def ms_pacman_reset_post_process(data):
    data["lives"] = 3


def normalize_ram(x):
    return x / 255


tenv = gym.make("MsPacman-ram-v0", frameskip=15)

state_count=tenv.observation_space.shape[0]
#state_count = tenv.observation_space.n
action_count=tenv.action_space.n

save_file = "saves/pacman_formal"
print("state size: {}, action size: {}".format(state_count, action_count))


#dqnagent = DQNAgent(state_count, action_count, discount_factor=0.95, learning_rate=0.0002,
#                    memory_size=10**6, hidden_layers=[256,256])

dqnagent = DQNAgent.load(save_file)
dqnagent.learning_rate = 0.0001
dqnagent.train(tenv, episode_count=100000,
                epsilon=0.1, epsilon_min=0.1, epsilon_decay=0.997, decay_epsilon_on_episodes=True,
                batch_size=32,
                replay_size_required=500,
                visualize=True, verbose=2,
                save_file=save_file, save_interval=25,
                init_postprocess_data_fun=mspacman_init_data, reset_postprocess_fun=ms_pacman_reset_post_process,
                step_postprocess_fun=ms_pacman_step_post_process,
                state_preprocess_fun=normalize_ram)
