#!/usr/bin/env python

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DoubleDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = []
        self.batch_size = 32
        self.model = self._build_model()
        self.target_model = self._build_model()
        
    def _build_model(self):
        # Deep neural network with 2 hidden layers of size 24 each
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])
        
        # Use the main network to select actions and the target network to evaluate them
        target_actions = np.argmax(self.model.predict(next_states), axis=1)
        target_q_values = self.target_model.predict(next_states)
        target_q_values = rewards + self.gamma * target_q_values[np.arange(len(target_q_values)), target_actions] * (1 - dones)
        
        # Update the Q-values for the selected actions in the main network
        q_values = self.model.predict(states)
        q_values[np.arange(len(q_values)), actions] = target_q_values
        
        # Train the main network on the updated Q-values
        self.model.fit(states, q_values, epochs=1, verbose=0)
        
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
# TODO:
# add main     

import time
import rospy
import random
import numpy as np
from rotate import Env
from collections import deque
from keras.optimizers import RMSprop
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation


EPISODES = 2000

class DDQN():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.gamma = 0.99 # discount factor        
        self.epsilon = 1.0 # initial exploration rate
        self.epsilon_min = 0.05 # minimum exploration rate    
        self.epsilon_decay = 0.99 # exploration rate decay factor
        self.batch_size = 64 # batch size for training
        self.train_start = 64
        self.target_update = 2000
        self.learning_rate = 0.00025
        self.memory = deque(maxlen=1000000) # replay buffer for experience replay

        self.model = self.buildModel(state_size, action_size)
        self.target_model = self.buildModel(state_size, action_size)

        self.updateTargetModel()
    
    def buildModel(self, state_size, action_size):
        model = Sequential()
        dropout = 0.2

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))

        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model        
    
    #qlearn function
    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.gamma * np.amax(next_target)
    
    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())
    
    #qlearn function
    def getAction(self, state):

        #if random number is less than epsilon choose random action to explore
        #else use model to predict q value and return action with maximum q value
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            action=random.randrange(self.action_size)
            # print("Random action:")
            return action
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            # print("Non random action:")
            return np.argmax(q_value[0])
        
    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))
            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))
            

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == '__main__':
    rospy.init_node('minDQN')

    state_size = 35
    action_size = 3

    env = Env()

    agent = DDQN(state_size, action_size)
    # scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    for e in range(1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        for t in range(agent.episode_step): 
            action = agent.getAction(state)

            next_state, reward, done = env.step(action)

            agent.appendMemory(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)
            
            score += reward
            state = next_state

            if t >= 750:
                print("REACHED 750 STEPS")
                done = True
            
            if done:
                agent.updateTargetModel()
                # scores
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)                
            
                print("Ep: %d score: %f epsilon: %.2f time: %d:%02d:%02d"%(e, score, agent.epsilon, h, m, s))
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                print("UPDATE TARGET NETWORK")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay        
'''
NOTES:
    - epsilon decay at the end of every episode too slow
    - crash detection not consistent
    - noise in lidar sensor?
    -
'''
