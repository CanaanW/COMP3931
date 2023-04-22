#! /usr/bin/env python

import rospy
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import math
from math import pi
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import random
import csv

class Env():
    def __init__(self):
        rospy.Subscriber('/odom', Odometry, self.odom)
        rospy.Subscriber('/scan', LaserScan, self.scan)
        self.reset_world = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.rate = rospy.Rate(10)
        self.goal = [2,2]
        self.position = []
        self.rot = Twist()
        self.crash = False
        self.goal_angle = 0

    def get_goal_distance(self, p1):
        return np.sqrt((self.goal[0] - p1[0])**2 + (self.goal[1]-p1[1])**2)

    def odom(self, odom):  
        self.position = [odom.pose.pose.position.x, odom.pose.pose.position.y]
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]

        _, _, yaw = euler_from_quaternion(orientation_list)
        
        goal_line = math.atan2(self.goal[1] - self.position[1], self.goal[0] - self.position[0])
        goal_angle = goal_line - yaw
        if goal_angle > 2*pi:
            goal_angle -= 2*pi
        elif goal_angle < -2*pi:
            goal_angle += 2*pi
        self.goal_angle = round(goal_angle, 2)
    
    def scan(self, scan):
        scan_range = []

        for i in scan.ranges:
            if i == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(i):
                scan_range.append(0)
            else:
                scan_range.append(i)

        self.scan_range = scan_range

    def move_left(self):
        self.rot.linear.x = 0
        self.rot.angular.z = 1

    def move_right(self):
        self.rot.linear.x = 0
        self.rot.angular.z = -1

    def move_forward(self):
        self.rot.angular.z = 0
        self.rot.linear.x = 0.15

    def get_reward(self, state, crash):
        reward = 0

        prev_distance = round(state[-1],2)
        curr_distance = round(state[-2],2)
        goal_angle = state[-3]
        
        if prev_distance>curr_distance:
            distance_reward = 1
        else:
            distance_reward = -1
        
        if curr_distance < 0.15:
            print("---GOAL FOUND!---\n"
                  "---GOAL FOUND!---\n"
                  "---GOAL_FOUND!---")
            reward = 500
        elif crash:
            print("\n---COLLISION!---\n")
            reward = -500
        else:
            reward = (prev_distance-curr_distance)+(distance_reward*10)+(-goal_angle**2)

            with open('rewards.txt', mode = 'a') as csv_file:
                reward_writer = csv.writer(csv_file, delimiter=",")
                reward_writer.writerow([reward])
        return reward
        
    def get_state(self):
        crash=False
        prev_distance = self.curr_distance
        self.curr_distance = self.get_goal_distance(self.position)

        goal_angle = self.goal_angle

        min_abs_distance = min(self.scan_range)
        if 0<min_abs_distance<0.12:
            crash = True
        
        return self.scan_range + [goal_angle, self.curr_distance, prev_distance], crash
    
    def step(self, action):
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_forward()
        elif action == 2:
            self.move_right()
        self.pub.publish(self.rot)

        state, crash = self.get_state()
        reward = self.get_reward(state, crash)

        return np.asarray(state), reward, crash
    
    def reset(self):
        with open ('rewards.txt', mode ='a') as csv_file:
            csv_file.write("--------------------------------------")

        self.prev_distance = self.get_goal_distance(self.position)
        self.curr_distance = self.get_goal_distance(self.position)
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_world()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        
        state, _ = self.get_state()

        return np.asarray(state)
# import rospy
# from geometry_msgs.msg import Twist
# from sensor_msgs.msg import LaserScan
# from gazebo_msgs.msg import ModelStates
# from nav_msgs.msg import Odometry
# from std_srvs.srv import Empty
# import math
# from math import pi
# import numpy as np
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
# import random

# class QLearn:
#     def __init__(self, actions, epsilon, alpha, gamma):
#         self.q = {}
#         self.epsilon = epsilon  # exploration constant
#         self.alpha = alpha      # discount constant
#         self.gamma = gamma      # discount factor
#         self.actions = actions
#         # self.goal=(2,2)

#     def getQ(self, state, action):
#         # print("Q(%s,%s) = %f"%(state, action, self.q.get((state, action),0.0)))
#         return self.q.get((state, action), 0.0)

#     def learnQ(self, state, action, reward, value):
#         '''
#         Q-learning:
#             Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a)))        
#         '''
#         oldv = self.q.get((state, action), None)
#         if oldv is None:
#             self.q[(state, action)] = reward
#         else:
#             self.q[(state, action)] = oldv + self.alpha * (value - oldv)

#     def chooseAction(self, state, return_q=False):
#         q = [self.getQ(state, a) for a in self.actions]
#         maxQ = max(q)

#         if random.random() < self.epsilon:
#             minQ = min(q); mag = max(abs(minQ), abs(maxQ))
#             # add random values to all the actions, recalculate maxQ
#             q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
#             maxQ = max(q)
#             # print("maxQ=%f"%maxQ)

#         count = q.count(maxQ)
#         # In case there're several state-action max values 
#         # we select a random one among them
#         if count > 1:
#             best = [i for i in range(len(self.actions)) if q[i] == maxQ]
#             i = random.choice(best)
#         else:
#             i = q.index(maxQ)

#         action = self.actions[i]        
#         if return_q: # if they want it, give it!
#             return action, q
#         return action

#     def learn(self, state1, action1, reward, state2):
#         maxqnew = max([self.getQ(state2, a) for a in self.actions])
#         self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)


# class AvoidObstacle(QLearn):
#     def __init__(self):
#         rospy.init_node('rotate_turtlebot')
#         rospy.Subscriber('/scan', LaserScan, self.scan)
#         # rospy.Subscriber('/gazebo/model_states', ModelStates, self.pose)
#         rospy.Subscriber('/odom', Odometry, self.odom)
#         self.reset = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
#         self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
#         self.rate = rospy.Rate(10)
#         self.goal = [2,2]
#         self.position = []
#         self.rot = Twist()

#         # Define the actions that can be taken by the turtlebot
#         self.actions = ['left', 'right', 'forward']

#         # Initialize QLearn
#         self.qlearn = QLearn(self.actions, epsilon=1, alpha=0.1, gamma=0.4)

#         # Set initial state
#         self.state = 'no_obstacle'

#     def get_goal_distance(self, p1):
#         return np.sqrt((self.goal[0] - p1[0])**2 + (self.goal[1]-p1[1])**2)

#     def odom(self, odom):  
#         self.position = [odom.pose.pose.position.x, odom.pose.pose.position.y]
        
#     def move_left(self):
#         self.rot.linear.x = 0
#         self.rot.angular.z = 1

#     def move_right(self):
#         self.rot.linear.x = 0
#         self.rot.angular.z = -1

#     def move_forward(self):
#         self.rot.angular.z = 0
#         self.rot.linear.x = 0.3

#     def get_reward(self, state):
#         #cumulative reward??
#         reward = 0
#         if state == 'obstacle_front':
#             reward = -10
#         elif state == 'obstacle_left':
#             reward = -5
#         elif state == 'obstacle_right':
#             reward = -5
#         elif state == 'closer':
#             reward = 3
#         elif state == 'farther':
#             reward = -3
#         elif self.goal_distance < 0.15:
#             reward = 200
#         elif state == 'crash':
#             reward = -200
#         return reward

        
#     def scan(self, scan):
#         # Get turtlebot x,y coords
#         previous_pos = self.position

#         # Wait then get NEW turtlebot x,y coords 
#         rospy.sleep(0.05)
#         current_pos = self.position

#         # Calculate previous distance and current distance (current distance should be smaller than previous distance)
#         previous_distance = self.get_goal_distance(previous_pos)
#         self.goal_distance = self.get_goal_distance(current_pos)
#         print("goal distance = %f"%self.goal_distance)

#         # Find closest object to turtlebot in front, left and right
#         front_min_distance = min(scan.ranges[1:10] + scan.ranges[-10:])
#         left_min_distance = min(scan.ranges[11:30])
#         right_min_distance = min(scan.ranges[-30:-11])

#         abs_min_distance = min(scan.ranges)

#         # Set distance thresholds for object proximity
#         check_distance = 0.6
#         side_check_distance = check_distance - 0.1

#         # If the nearest object to turtlebot is closer than threshold then there is an object, change state to reflect this
#         if front_min_distance < check_distance:
#             self.state = 'obstacle_front'
#         else:
#             if left_min_distance < right_min_distance and left_min_distance < side_check_distance:
#                 self.state = 'obstacle_left'
#             elif right_min_distance < left_min_distance and right_min_distance < side_check_distance:
#                 self.state = 'obstacle_right'
#             else:
#                 if self.goal_distance < previous_distance:
#                     if self.heading>0 and self.heading<1:
#                         self.state = 'closer + 1'
#                     else:
#                         self.state = 'closer'
#                 else:
#                     if self.heading>0 and self.heading<1:
#                         self.state = 'farther + 1'
#                     else:
#                         self.state = 'farther'
        
#         if abs_min_distance < 0.15:
#             self.state = 'crash'
#             print("YOU CRASHED!!")
#             self.reset()

#         # Get the reward for the current state
#         reward = self.get_reward(self.state)

#         # Choose an action based on the current state
#         action = self.qlearn.chooseAction(self.state)
#         print(self.state, action)

#         # Perform action chosen by q-learning algorithm 
#         if action == self.actions[0]:
#             self.move_left()
#         elif action == self.actions[1]:
#             self.move_right()
#         elif action == self.actions[2]:
#             self.move_forward()
#         self.pub.publish(self.rot)

#         # Learn from the experience
#         self.qlearn.learn(self.state, action, reward, self.state)

#         # Sleep for a short period to control the frequency of the loop
#         self.rate.sleep()

# if __name__ == "__main__":
#     try:
#         AvoidObstacle()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass