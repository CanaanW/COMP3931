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
        self.rewards_episode = [0]
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

        # self.rate.sleep()
    
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
        # self.rate.sleep()


    def move_left(self):
        self.rot.linear.x = 0
        self.rot.angular.z = 1

    def move_right(self):
        self.rot.linear.x = 0
        self.rot.angular.z = -1

    def move_forward(self):
        self.rot.angular.z = 0
        self.rot.linear.x = 0.15

    def stop(self):
        self.rot.angular.z = 0
        self.rot.linear.x = 0

    def get_reward(self, state, crash):
        reward = 0

        prev_distance = round(state[-1],2)
        curr_distance = round(state[-2],2)
        goal_angle = state[-3]
        
        if prev_distance>curr_distance:
            distance_reward = 10
        elif prev_distance == curr_distance:
            distance_reward = 0
        else:
            distance_reward = -10
        
        if curr_distance < 0.15:
            print("---GOAL FOUND!---\n"
                  "---GOAL FOUND!---\n"
                  "---GOAL_FOUND!---")
            reward = 500
        elif crash:
            print("\n---COLLISION!---\n")
            self.stop()
            reward = -500
        else:
            reward = (prev_distance-curr_distance)+(distance_reward)+(-goal_angle**2)
        with open('rewards.txt', mode = 'a') as csv_file:
            reward_writer = csv.writer(csv_file, delimiter=",")
            reward_writer.writerow([reward])
            
        self.rewards_episode.append(reward)
        return reward
        
    def get_state(self):
        crash=False
        prev_distance = self.curr_distance
        self.curr_distance = self.get_goal_distance(self.position)

        goal_angle = self.goal_angle

        min_abs_distance = min(self.scan_range)
        if 0<min_abs_distance<0.12:
            # print(min_abs_distance)
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
        # self.rate.sleep()


        state, crash = self.get_state()
        # print(crash)
        reward = self.get_reward(state, crash)

        return np.asarray(state), reward, crash
    
    def reset(self):
        mean_reward = float(np.mean(self.rewards_episode))
        self.rewards_episode = []
        with open ('rewards.txt', mode ='a') as csv_file:
            csv_file.write("mean reward for episode: %f \n"%mean_reward)
            csv_file.write("--------------------------------------\n")        

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_world()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        
        self.prev_distance = self.get_goal_distance([-2,-2])
        self.curr_distance = self.get_goal_distance([-2,-2])
        
        state, crash = self.get_state()

        return np.asarray(state)