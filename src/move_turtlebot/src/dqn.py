#!/usr/bin/env python

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import rospy
from rotate import Env
from collections import deque


class DoubleDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = []
        self.batch_size = 64
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

if __name__ == '__main__':
    rospy.init_node('DDQN')

    state_size = 35
    action_size = 3

    env = Env()

    agent = DoubleDQNAgent(state_size, action_size)
    global_step = 0
    start_time = time.time()

    for e in range(1, 20000):
        done = False
        state = env.reset()
        score = 0
        for t in range(751): 
            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            score += reward
            state = next_state

            if t >= 750:
                print("REACHED 750 STEPS")
                done = True
            
            if done:
                agent.update_target_network()
                # scores
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)                
            
                print("Ep: %d score: %f epsilon: %.2f time: %d:%02d:%02d"%(e, score, agent.epsilon, h, m, s))
                break

            global_step += 1
            if global_step % 2000 == 0:
                print("UPDATE TARGET NETWORK")

        if agent.epsilon > agent.epsilon_min:
            agent.decay_epsilon()