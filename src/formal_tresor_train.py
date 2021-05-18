import gym
import gym_tresor2d
from agents.qlearning_agent import QLearningAgent

save_file="saves/tresor_formal"

env = gym.make("tresor2d-v0", width=10, height=10, generation="zigzag")
state_count=env.observation_space.n
action_count=env.action_space.n

print("=====ENTRAINEMENT D'AGENT SUR RECHERCHE DE TRESOR AVEC Q-LEARNING=======")

agent = QLearningAgent(state_count, action_count, 0.98, 0.1)

#Uncomment to resume training from previous file
#agent = QLearningAgent.load(save_file)

agent.train(env, episode_count=8000, 
            epsilon=1.0, epsilon_decay=0.97, epsilon_min=0.001, decay_epsilon_on_episodes=True,
            visualize=False, save_file=save_file, verbose=2)