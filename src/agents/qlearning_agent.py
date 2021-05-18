import pandas as pd
import numpy as np
import random
import pickle
import time
import os

class QLearningAgent():
    QLAGENT_FILE_SUFFIX = ".qlearn"


    def __init__(self, state_count, action_count, discount_factor=0.5, learning_rate=0.1):
        self.state_count = state_count
        self.action_count = action_count
        self.qtable = self._build_table()

        self.discount_factor = discount_factor    #discount rate
        self.learning_rate = learning_rate     #learning rate


    #Create the q-table that stores the AI data
    def _build_table(self):
        return np.zeros([self.state_count, self.action_count])


    def act(self, state, epsilon=0.0):
        """Choose an action to do according to the q-table, taking into account the epsilon greedy policy"""
        if epsilon > 0.0 and np.random.rand() <= epsilon:
            return random.randrange(self.action_count)  #returns random action
        else:
            return np.argmax(self.qtable[state])  # returns best action
    

    #Train the model to fit the given environment
    def train(self, env, episode_count=1000, visualize=False, 
            epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.0, decay_epsilon_on_episodes=True,
            save_file="", save_interval=25,
            target_fps=-1,verbose=2, train=True):


        rewards_history = []
        steps_per_episode_history = []
        epsilon_history = []
        
        for episode in range(episode_count):
            steps_since_last_reset = 0
            reward_since_last_reset = 0
            done = False

            state = env.reset()
            while not done:
                #---Act
                chosen_action = self.act(state, epsilon)
                new_state, reward, done, _ = env.step(chosen_action)

                #---Learn
                if train:
                    #Get predicted reward from new_state
                    q_predict = self.qtable[state, chosen_action]

                    #Get desired reward from new_state
                    if done: 
                        q_target = reward     
                    else:
                        q_target = reward + self.discount_factor * np.max(self.qtable[new_state])
                        
                    #Adjust the value
                    self.qtable[state, chosen_action] = (1-self.learning_rate) * self.qtable[state, chosen_action] + self.learning_rate * (q_target - q_predict)  # update

                state = new_state
                reward_since_last_reset += reward

                #Decay epsilon if we're decaying per step
                if not decay_epsilon_on_episodes:
                    epsilon = max(epsilon_min, epsilon*epsilon_decay)
  
                steps_since_last_reset += 1

                if verbose==2:
                    print("Step: {}; reward={}; epsilon:{}".format(steps_since_last_reset,reward_since_last_reset,epsilon),end="\r")


                if verbose >= 3:
                    print("State: {}=\n{}; reward={}; epsilon={}".format(state, self.qtable[state, :],reward,epsilon))
                if visualize == True:
                    env.render()
                    if target_fps > 0:
                        time.sleep(1.0/target_fps)

            #---episode ended
            
            if verbose >= 1:
                print("Episode {} done in {} steps with reward {}, eps={}".format(episode, steps_since_last_reset, reward_since_last_reset, epsilon))
            
            if save_file!="" and episode%save_interval==0:
                self.save(save_file)

            #decay epsilon if we're decaying per episode
            if decay_epsilon_on_episodes:
                epsilon = max(epsilon_min, epsilon*epsilon_decay)

            rewards_history.append(reward_since_last_reset)
            epsilon_history.append(epsilon)
            steps_per_episode_history.append(steps_since_last_reset)
        return steps_per_episode_history, rewards_history, epsilon_history

            



    def get_written_summary(self):
        res = "- learning rate: {}\n".format(self.learning_rate)
        res += "- discount factor: {}\n".format(self.discount_factor)
        res += "- action count: {}\n".format(self.action_count)
        res += "- state count: {}\n".format(self.state_count)
        return res


    def save(self, path):
        """Save the Agent"""
        if not path.endswith(QLearningAgent.QLAGENT_FILE_SUFFIX):
            path += QLearningAgent.QLAGENT_FILE_SUFFIX
        try:
            pickle.dump(self, open(path,"wb"))
            print("Q-Learning Agent successfully saved")
        except IOError:
            print("Error saving the agent")    


    @staticmethod
    def load(path):
        """Load the DQN Agent"""
        if not path.endswith(QLearningAgent.QLAGENT_FILE_SUFFIX) and not os.path.isfile(path):
            path += QLearningAgent.QLAGENT_FILE_SUFFIX
        try:
            agent = pickle.load(open(path, "rb"))
            print("Q-Learning Agent successfully loaded")
            return agent
        except IOError:
            print("No existing agent found")