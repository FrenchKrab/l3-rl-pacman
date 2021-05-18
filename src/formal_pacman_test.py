import gym
from agents.dqn_agent_v2 import DQNAgent
from misc.pacman_tools import *


tenv = gym.make("MsPacman-ram-v0", frameskip=15)

state_count=tenv.observation_space.shape[0]
action_count=tenv.action_space.n

save_file = "saves/pacman_formal.dqn"
print("=====TEST D'AGENT SUR PACMAN AVEC DQN=======")


#dqnagent = DQNAgent(state_count, action_count, discount_factor=0.95, learning_rate=0.0002,
#                    memory_size=10**6, hidden_layers=[256,256])

dqnagent = DQNAgent.load(save_file)

#hyperparamètres d'entrainement: epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.997, decay_epsilon_on_episodes=True
#batch_size=32, replay_size_required=500
#Note: n'hésitez pas à augmenter target_fps, ou à meme le passer à -1 afin d'aller le plus vite possible et juste lire les scores.
dqnagent.train(tenv, episode_count=100,
                epsilon=0.05, epsilon_min=0.05, epsilon_decay=1.0, decay_epsilon_on_episodes=True,
                batch_size=32,
                replay_size_required=500,
                visualize=True, verbose=2,
                save_file="", save_interval=25,
                init_postprocess_data_fun=mspacman_init_data, reset_postprocess_fun=ms_pacman_reset_post_process,
                step_postprocess_fun=ms_pacman_step_post_process,
                state_preprocess_fun=normalize_ram, train=False, target_fps=10)

