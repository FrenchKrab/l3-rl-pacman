from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import time

from keras.models import load_model
from pathlib import Path
import pickle

import random
import numpy as np



class DQNAgent():
    NEURAL_NETWORK_FILE_SUFFIX = "_nn.h5"
    REPLAY_FILE_SUFFIX = "_replay.dat"
    DQN_FILE_SUFFIX = ".dqn"
    
    def __init__(self, input_size, action_size, discount_factor=0.95, learning_rate=0.001,
                memory_size=100000,
                hidden_layers=[], activation_function="relu", loss_function="mse"):

        #Input and output layers size
        self.input_size = input_size
        self.action_size = action_size

        #Replay memory
        self.memory = deque(maxlen=memory_size)


        #Hyperparameters
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        #Model building data
        self.hidden_layers = hidden_layers.copy()
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.model = self._build_model(hidden_layers.copy(), activation_function, loss_function)



    def _build_model(self, hidden_layers, activation_function, loss_function):
        """Build the neural network model according to the agent data"""
        
        model = Sequential()


        #If there's no hidden layer, directly plug the output layer to the inputs
        if len(self.hidden_layers) == 0:
            model.add(Dense(self.action_size, input_shape=(self.input_size,), activation="linear"))
        #Else, build with hidden layers
        else:
            #Add the first hidden layer (plugged to inputs)
            model.add(Dense(self.hidden_layers[0], input_shape=(self.input_size,), activation=self.activation_function))
            #add the rest of the hidden layers
            for i in range(1, len(self.hidden_layers)):
                model.add(Dense(self.hidden_layers[i], activation=self.activation_function))
            #add output layer
            model.add(Dense(self.action_size, activation="linear"))

        #Compile the model
        model.compile(loss=self.loss_function, optimizer=Adam(learning_rate=self.learning_rate))
                      #optimizer=RMSprop(learning_rate=self.learning_rate)) #alternative optimizer

        return model
    

    
    def memorize(self, state, action, reward, next_state, done):
        """Append (state, action, reward, next_state, done) to the memory """

        self.memory.append((state, action, reward, next_state, done))
    

    def act(self, state, epsilon=0.0):
        """Choose an action to do according to model, taking into account the epsilon greedy policy"""

        if epsilon > 0.0 and np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])  # returns action
    

    def learn_from_replay(self, batch_size, iteration_count=1):
        for _ in range(iteration_count):
            minibatch = random.sample(self.memory, batch_size)

            inputs = []
            target_outputs = []
            for state, action, reward, next_state, done in minibatch:
                target = reward #if done
                if not done:    #else
                    target = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target

                inputs.append(state[0])
                target_outputs.append(target_f[0])

            #Perform one train on batch
            self.model.train_on_batch(x=np.array(inputs), y=np.array(target_outputs))

    def train(self, env, episode_count=10000, batch_size=32, batch_iterations=1, replay_size_required=0,
            epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1, decay_epsilon_on_episodes=True,
            save_file="", save_interval=25,
            init_postprocess_data_fun=None, step_postprocess_fun=None, reset_postprocess_fun=None,
            state_preprocess_fun=None,
            verbose=2, visualize=True, train=True, target_fps=-1):
        """Start a training session of the DQN Agent

        - env: environnement to use
        - episode_count: number of episodes to train through
        - batch_size: size of a minibatch
        - batch_iterations: minibatch trainings per step
        - replay_size_required: minimum replay size required to start minibatch training
        - epsilon: initial epsilon value
        - epsilon_decay: value to multiply epsilon by to make it decrease
        - epsilon_min: minimal epsilon value possible
        - decay_epsilon_on_episodes: if true, epsilon will only decrease on episode completion, else, on each step
        - save_file: name of the file to save data to
        - save_interval: number of episode between each save
        - init_postprocess_data_fun: function that returns a desired object to store postprocessing data to
        - step_postprocess_fun: function that takes in (new_state, reward, done, infos, postprocess_data) and
                                outputs an postprocessed (new_state, reward, done, infos, postprocess_data)
        - reset_postprocess_fun: function called after each environnement reset, takes in custom postprocess_data
        - state_preprocess_fun: function that takes in a state and output a preprocessed state
        - train: if True, the agent will learn; else it won't
        - target_fps: desired game speed. Set to -1 for maximum speed
        """

        rewards_history = []
        steps_per_episode_history = []
        epsilon_history = []


        #Initialize postprocess data if possible
        if init_postprocess_data_fun!=None:
            postprocess_data = init_postprocess_data_fun()
        
        for episode in range(episode_count):
            #---Episode starting
            #Meta informations
            steps_since_last_reset = 0
            reward_since_last_reset = 0

            #---Reset state
            state = env.reset()
            #Preprocess the state
            if state_preprocess_fun != None:
                state = state_preprocess_fun(state)
            state = np.reshape(state, [1,self.input_size])
            #Call post process function after reset
            if reset_postprocess_fun!=None:
                reset_postprocess_fun(postprocess_data)
            
            done = False    #Is the episode over

            #Repeat while episode isn't over (episode main loop)
            while not done:
                #---Act
                chosen_action = self.act(state, epsilon)
                new_state, reward, done, infos = env.step(chosen_action)
                
                if state_preprocess_fun != None:
                    new_state = state_preprocess_fun(new_state)

                if step_postprocess_fun!=None:
                    new_state, reward, done, infos = step_postprocess_fun(new_state, reward, done, infos, postprocess_data)

                new_state = np.reshape(new_state, [1,self.input_size])
                if train:
                    self.memorize(state, chosen_action, reward, new_state, done)
                state = new_state


                #---Learn
                if train and len(self.memory) > batch_size and len(self.memory) > replay_size_required:
                    self.learn_from_replay(batch_size, batch_iterations)
                    #Decay epsilon if we're decaying per step
                    if not decay_epsilon_on_episodes:
                        epsilon = max(epsilon_min, epsilon*epsilon_decay)

                #---Wait for target fps
                if target_fps > 0:
                    time.sleep(1.0/target_fps)


                #---Debug and visualize
                steps_since_last_reset += 1
                reward_since_last_reset += reward

                if verbose == 2:
                    print("Step: {}; reward={}; epsilon:{}".format(steps_since_last_reset,reward_since_last_reset,epsilon),end="\r")
                elif verbose >= 3:
                    print("Step: {}; reward={}; epsilon:{}".format(steps_since_last_reset,reward,epsilon))

                if visualize == True:
                    env.render()
            
            #---Episode ended

            if verbose >= 1:
                print("Episode {} done in {} steps with total reward {}, e={}".format(episode, steps_since_last_reset, reward_since_last_reset, epsilon))

            #decay epsilon if we're decaying per episode
            if decay_epsilon_on_episodes and len(self.memory) > replay_size_required and train:
                epsilon = max(epsilon_min, epsilon*epsilon_decay)
            
            #Save interval
            if save_file != "" and episode%save_interval == 0:
                self.save(save_file)

            steps_per_episode_history.append(steps_since_last_reset)
            rewards_history.append(reward_since_last_reset)
            epsilon_history.append(epsilon)
        return steps_per_episode_history, rewards_history, epsilon_history

    def get_written_summary(self):
        res = "- learning rate: {}\n".format(self.learning_rate)
        res += "- discount factor: {}\n".format(self.discount_factor)
        res += "- activation function: {}\n".format(self.activation_function)
        res += "- loss function: {}\n".format(self.loss_function)
        res += "- experience replay capacity: {}\n".format(self.memory.maxlen)
        res += "- experience replay size: {}\n".format(len(self.memory))
        res += "Neural network architecture:\n"
        res += "Input({}) -> ".format(self.input_size)
        for neuron_count in self.hidden_layers:
            res += "Dense({}) -> ".format(neuron_count)
        res += "Output({})".format(self.action_size)
        return res

    #--------SAVE/LOAD RELATED---------

    def save(self, path):
        """Save the DQN Agent"""
        if not path.endswith(DQNAgent.DQN_FILE_SUFFIX):
            path += DQNAgent.DQN_FILE_SUFFIX
        try:
            pickle.dump(self, open(path,"wb"))
            print("DQN Agent successfully saved")
        except IOError:
            print("Error saving the agent")    


    @staticmethod
    def load(path):
        """Load the DQN Agent"""
        try:
            agent = pickle.load(open(path, "rb"))
            print("DQN Agent successfully loaded")
            return agent
        except IOError:
            print("No existing agent found")




    def save_all_individually(self, path):
        """Save the DQN Agent's components (neural network model, replay memory)"""

        self.save_model(path+DQNAgent.NEURAL_NETWORK_FILE_SUFFIX)
        self.save_replay(path+DQNAgent.REPLAY_FILE_SUFFIX)
        #keras.utils.plot_model(self.model, to_file='_preview.png')
    

    def load_all_individually(self, path):
        """Load the DQN Agent's components (neural network model, replay memory)"""

        self.load_model(path+DQNAgent.NEURAL_NETWORK_FILE_SUFFIX)
        self.load_replay(path+DQNAgent.REPLAY_FILE_SUFFIX)


    def save_model(self, path):
        """Save only the neural network model"""

        try:
            self.model.save(path)
            print("Neural network successfully saved")
        except IOError:
            print("Error saving the model")


    def save_replay(self, path):
        """Save only the replay memory"""

        try:
            pickle.dump(self.memory, open(path,"wb"))
            print("Replay successfully saved")
        except IOError:
            print("Error saving the replay")    
    

    def load_model(self, path):
        """Load only the neural network model"""

        try:
            model = load_model(path)
            self.model = model
            print("Model successfully loaded")
        except IOError:
            print("No existing model found")
        except ImportError:
            print("Existing model corrupted")
    

    def load_replay(self, path):
        """Load only the replay memory"""

        try:
            memory = pickle.load(open(path, "rb"))
            self.memory = memory
            print("Replay memory successfully loaded")
        except IOError:
            print("No existing replay found")
