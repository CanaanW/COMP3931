#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
# from math import dist
import numpy as np
import random

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        # self.goal=(2,2)

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)


class AvoidObstacle(QLearn):
    def __init__(self):
        rospy.init_node('rotate_turtlebot')
        rospy.Subscriber('/scan', LaserScan, self.scan)
        # rospy.Subscriber('/gazebo/model_states', ModelStates, self.pose)
        rospy.Subscriber('/odom', Odometry, self.odom)
        self.reset = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.rate = rospy.Rate(10)
        self.goal = [2,2]
        self.position = []
        self.rot = Twist()

        # Define the actions that can be taken by the turtlebot
        self.actions = ['left', 'right', 'forward']

        # Initialize QLearn
        self.qlearn = QLearn(self.actions, epsilon=0.9, alpha=0.1, gamma=0.7)

        # Set initial state
        self.state = 'no_obstacle'

    def get_goal_distance(self, p1):
        return np.sqrt((self.goal[0] - p1[0])**2 + (self.goal[1]-p1[1])**2)

    def odom(self, data):  
        self.position = [data.pose.pose.position.x, data.pose.pose.position.y]
    
    def stop(self):
        self.rot.linear.x = 0
        self.rot.angular.z = 0

    def move_left(self):
        self.rot.linear.x = 0
        self.rot.angular.z = 1

    def move_right(self):
        self.rot.linear.x = 0
        self.rot.angular.z = -1

    def move_forward(self):
        self.rot.angular.z = 0
        self.rot.linear.x = 0.3

    def get_reward(self, state):
        #cumulative reward??
        reward = 0
        if state == 'obstacle_front':
            reward = -10
        elif state == 'obstacle_left':
            reward  = -5
        elif state == 'obstacle_right':
            reward = -5
        elif state == 'closer':
            reward = 3
        elif state == 'farther':
            reward = -3
        elif self.current_distance < 0.1:
            reward = 100
        elif state == 'crash':
            reward = -100
        else:
            reward = 1
        return reward

        
    def scan(self, data):
        # Get turtlebot x,y coords
        previous_pos = self.position

        # Wait for half a second then get NEW turtlebot x,y coords 
        rospy.sleep(0.5)
        current_pos = self.position

        # Calculate previous distance and current distance (current distance should be smaller than previous distance)
        previous_distance = self.get_goal_distance(previous_pos)
        self.current_distance = self.get_goal_distance(current_pos)

        # Find closest object to turtlebot in front, left and right
        front_min_distance = min(data.ranges[1:10] + data.ranges[-10:])
        left_min_distance = min(data.ranges[11:30])
        right_min_distance = min(data.ranges[-30:-11])

        abs_min_distance = min(data.ranges)

        # Set distance thresholds for object proximity
        check_distance = 0.5
        side_check_distance = check_distance - 0.1

        # If the nearest object to turtlebot is closer than threshold then there is an object, change state to reflect this
        if front_min_distance < check_distance:
            self.state = 'obstacle_front'
        else:
            if left_min_distance < right_min_distance and left_min_distance < side_check_distance:
                self.state = 'obstacle_left'
            elif right_min_distance < left_min_distance and right_min_distance < side_check_distance:
                self.state = 'obstacle_right'
            else:
                if self.current_distance < previous_distance:
                    self.state = 'closer'
                else:
                    self.state = 'farther'
        
        if abs_min_distance < 0.2:
            self.state = 'crash'
            print("YOU CRASHED!!")
            self.reset()

        # Get the reward for the current state
        reward = self.get_reward(self.state)

        # Choose an action based on the current state
        action = self.qlearn.chooseAction(self.state)
        print(self.state, action)

        # Perform action chosen by q-learning algorithm 
        if action == self.actions[0]:
            self.move_left()
        elif action == self.actions[1]:
            self.move_right()
        elif action == self.actions[2]:
            self.move_forward()
        self.pub.publish(self.rot)

        # Learn from the experience
        self.qlearn.learn(self.state, action, reward, self.state)

        # Sleep for a short period to control the frequency of the loop
        self.rate.sleep()

if __name__ == "__main__":
    try:
        AvoidObstacle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass