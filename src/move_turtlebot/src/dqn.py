#!/usr/bin/env python

import random
import numpy as np
import csv
import time
import rospy
import random
from rotate import Env
from collections import deque
from keras.optimizers import RMSprop
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation

class DQN():
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

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()
    
    def buildModel(self):
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

EPISODES = 2000

if __name__ == '__main__':
    rospy.init_node('DDQN')

    state_size = 35
    action_size = 3

    env = Env()

    agent = DQN(state_size, action_size)
    global_step = 0
    start_time = time.time()

    for e in range(1, EPISODES):
        done = False
        goal = False
        state,_ = env.reset()
        score = 0
        with open ("rewards.txt", mode='a') as file:
            file.write("episode %i: "%e)         
        for t in range(1000): 
            action = agent.getAction(state)

            next_state, reward, done, goal = env.step(action)

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

            global_step += 1
            if global_step % 2000 == 0:
                print("UPDATE TARGET NETWORK") 

            if goal:
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)  
                with open('rewards.txt', mode = 'a') as file:
                    file.write("\nEp: %d score: %f epsilon: %.2f time: %d:%02d:%02d"%(e, score, agent.epsilon, h, m, s))
                done = True                            
            
            if done:
                agent.updateTargetModel()
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)                
            
                print("Ep: %d score: %f epsilon: %.2f time: %d:%02d:%02d"%(e, score, agent.epsilon, h, m, s))
                break

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
