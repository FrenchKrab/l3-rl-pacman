import gym
import gym_tresor2d
from agents.qlearning_agent import QLearningAgent

save_file="saves/tresor_formal"

env = gym.make("tresor2d-v0", width=10, height=10, generation="zigzag")
state_count=env.observation_space.n
action_count=env.action_space.n

print("=====TEST D'AGENT SUR RECHERCHE DE TRESOR AVEC Q-LEARNING=======")

agent = QLearningAgent.load(save_file)
agent.train(env, episode_count=1000, epsilon=0.01, train=False,
            visualize=True, verbose=1, target_fps=10) 