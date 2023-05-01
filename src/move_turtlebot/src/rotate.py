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
from tf.transformations import euler_from_quaternion
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

    def stop(self):
        self.rot.angular.z = 0
        self.rot.linear.x = 0

    def get_reward(self, state, crash, goal):
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
        
        if goal:
            print("---GOAL FOUND!---")
            reward = 500
        elif crash:
            print("\n---COLLISION!---")
            self.stop()
            reward = -500
        else:
            reward = (prev_distance-curr_distance)+(distance_reward)+(-goal_angle**2)
        with open('rewards.txt', mode = 'a') as a:
            a.write(str(reward)+", ")
            
        self.rewards_episode.append(reward)
        return reward
        
    def get_state(self):
        crash = False
        goal = False
        prev_distance = self.curr_distance
        self.curr_distance = self.get_goal_distance(self.position)

        goal_angle = self.goal_angle

        min_abs_distance = min(self.scan_range)
        if 0<min_abs_distance<0.12:
            crash = True

        if self.curr_distance < 0.15:
            goal = True
        
        return self.scan_range + [goal_angle, self.curr_distance, prev_distance], crash, goal
    
    def step(self, action):
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_forward()
        elif action == 2:
            self.move_right()
        self.pub.publish(self.rot)

        state, crash, goal = self.get_state()
        reward = self.get_reward(state, crash, goal)

        return np.asarray(state), reward, crash, goal
    
    def reset(self):
        mean_reward = float(np.mean(self.rewards_episode))
        self.rewards_episode = [0]
        with open ('rewards.txt', mode ='a') as csv_file:
            csv_file.write("\nmean reward for episode: %f\n"%mean_reward)      

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_world()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        
        self.prev_distance = self.get_goal_distance([-2,-2])
        self.curr_distance = self.get_goal_distance([-2,-2])
        
        state, crash,_ = self.get_state()
        while crash:
            state,crash,_ = self.get_state()

        return (np.asarray(state), crash)