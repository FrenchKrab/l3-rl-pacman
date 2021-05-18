import pandas as pd
import numpy as np
import time
import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model
import keras.utils  #TEMP for preview model
import keras.callbacks
import datetime




class DQNAgent():
    def __init__(self, state_count, action_count, discount_factor=0.95, discrete_env=False, learning_rate=0.01, 
                epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.99, memory_size=1000, decay_epsilon_on_episodes=True,
                hidden_layers=[]):
        self.state_count = state_count
        self.action_count = action_count
        self.memory = deque(maxlen=memory_size)
        self.discrete_env = discrete_env
        self.hidden_layers = hidden_layers.copy()

        if discrete_env:
            self.input_layer_size = 2
        else:
            self.input_layer_size = self.state_count

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        #Used by policies
        self.training = False
        self.step = 0

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_epsilon_on_episodes = decay_epsilon_on_episodes
        self.model = self._build_model()


    #Build the model
    def _build_model(self):
        model = Sequential()
        layers = self.hidden_layers.copy()


        for i in range(0, len(layers)):
            if i == 0:
                #Create input layer
                model.add(Dense(layers[0], input_shape=(self.input_layer_size,), activation='relu'))
            else:
                model.add(Dense(layers[i], activation='relu'))

            #model.add(BatchNormalization())
            #model.add(LeakyReLU())
            #model.add(Dropout(0.1))

        model.add(Dense(self.action_count, activation="linear"))

        model.compile(loss='mse',
                      #optimizer=RMSprop(learning_rate=self.learning_rate))
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_count)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def save(self, path):
        try:
            self.model.save(path)
            print("File saved successfully")
        except IOError:
            print("Error saving the model")
        #keras.utils.plot_model(self.model, to_file='_preview.png')
        
    def load(self, path):
        try:
            model = load_model(path)
            self.model = model
            print("Model loaded successfully")
        except IOError:
            print("No existing model found")
        except ImportError:
            print("Existing model corrupted")

    def replay(self, batch_size, iteration_count=1):

        for _ in range(iteration_count):
            minibatch = random.sample(self.memory, batch_size)

            inputs = []
            target_outputs = []
            for state, action, reward, next_state, done in minibatch:
                #print("{},{},{},{}".format(state, action, reward, next_state))
                target = reward
                if not done:
                    target = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target

                inputs.append(state[0])
                target_outputs.append(target_f[0])
                #self.model.fit(state, target_f, epochs=1, verbose=0)


            #logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

            #self.model.fit(x=np.array(inputs), y=np.array(target_outputs), batch_size=batch_size, epochs=1, verbose=0,
             #               callbacks=[])
            self.model.train_on_batch(x=np.array(inputs), y=np.array(target_outputs))

        if not self.decay_epsilon_on_episodes and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    

    def fit(self, env, nb_steps=1000, visualize=False, visualize_interval=1, verbose=2,
            preprocess_fun=None,
            batch_size=32, batch_iterations=1, batch_step_interval=1, batch_size_start_learning=0,
            save_file="", save_interval=500,
            init_data_fun=None, step_post_process_fun=None, reset_post_process=None):
        self.training = True
        current_episode = 1
        steps_since_last_reset = 0
        reward_since_last_reset = 0

        if save_file != "":
            self.load(save_file)

        data={}
        if init_data_fun!=None:
            init_data_fun(data)

        #Reset to begin
        state = env.reset()
        if preprocess_fun != None:
            state = preprocess_fun(state)
        state = np.reshape(state, [1,self.input_layer_size])
        #Call post process function after reset
        if reset_post_process!=None:
            reset_post_process(data)
            
        for i in range(nb_steps):
            #---Act
            chosen_action = self.act(state)
            new_state, reward, done, infos = env.step(chosen_action)
            
            if preprocess_fun != None:
                new_state = preprocess_fun(new_state)

            if step_post_process_fun!=None:
                new_state, reward, done, infos = step_post_process_fun(new_state, reward, done, infos, data)


            new_state = np.reshape(new_state, [1,self.input_layer_size])
            self.memorize(state, chosen_action, reward, new_state, done)
            state = new_state

            #---Learn
            #temp condition with epsilon
            if (len(self.memory) > batch_size and len(self.memory) > batch_size_start_learning
                    and i%batch_step_interval==0):
                self.replay(batch_size, batch_iterations)


            #---Debug, print and visualize
            if done:
                if verbose >= 1:
                    print("Episode {} done in {} steps with reward {}, e={}".format(current_episode, steps_since_last_reset, reward_since_last_reset, self.epsilon))
                    if visualize == True:
                        env.render()
                        #time.sleep(0.0)
                steps_since_last_reset = 0
                reward_since_last_reset = 0
                current_episode += 1
            else:
                steps_since_last_reset += 1
                reward_since_last_reset += reward
                if verbose >= 2:
                    print("Step: {}; reward={}; epsilon:{}".format(i,reward,self.epsilon))
                if visualize == True and i%visualize_interval == 0:
                    env.render()
                    #time.sleep(0.01)
            
            #Save interval
            if save_file != "" and i%save_interval == 0:
                self.save(save_file)

            #---Reset the episode if on a terminal state
            if done:
                state = env.reset()
                if preprocess_fun != None:
                    state = preprocess_fun(state)
                state = np.reshape(state, [1,self.input_layer_size])
                #Call post process function after reset
                if reset_post_process!=None:
                    reset_post_process(data)

                #Decay the epsilon if we're decaying it on episode end
                if (self.decay_epsilon_on_episodes and self.epsilon > self.epsilon_min
                        and len(self.memory) > batch_size_start_learning):
                    self.epsilon *= self.epsilon_decay

    def test(self, env, episode_count, epsilon=0.05,
            init_data_fun=None, step_post_process_fun=None, reset_post_process=None):
        max_score = 0
        min_score = 0
        total_score = 0

        self.epsilon = epsilon

        for ep in range(episode_count):
            episode_score = 0

            #Reset to begin
            state = env.reset()
            if preprocess_fun != None:
                state = preprocess_fun(state)

            data={}
            if reset_post_process!=None:
                reset_post_process(data)

            done = False
            while not done:
                chosen_action = self.act(state)
                new_state, reward, done, infos = env.step(chosen_action)
                
                if preprocess_fun != None:
                    new_state = preprocess_fun(new_state)

                if step_post_process_fun!=None:
                    new_state, reward, done, infos = step_post_process_fun(new_state, reward, done, infos, data)
                
                state = new_state

                
                episode_score += reward
            
            max_score = max(max_score, episode_score)
            min_score = min(min_score, episode_score)
            total_score += episode_score
            
        average_score = total_score / episode_count

        return min_score, max_score, average_score