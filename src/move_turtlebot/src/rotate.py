#! /usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import gym
import rospkg
# from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import random

'''
Actions = [0, 1, 2]
move_left = 0
move_right = 1
move_forward = 2

State = [x,y, theta, front_min, left_min, right_min]
'''
class MyEnv(gym.Env):
    def __init__(self):

        self.action_space = spaces.Discrete(3)
        # self.observation_space = 


class QLearn(object):
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

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
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.pose)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)
        self.rate = rospy.Rate(10)
        self.rot = Twist()

    def stop(self):
        self.rot.linear.x=0
        self.rot.angular.z=0

    def move_left(self):
        self.rot.linear.x = 0
        self.rot.angular.z = 1

    def move_right(self):
        self.rot.linear.x = 0
        self.rot.angular.z = -1

    def move_forward(self):
        self.rot.angular.z = 0
        self.rot.linear.x = 0.3

    def move_back(self):
        self.rot.angular.z = 0
        self.rot.linear.x = -0.3

    def pose(self, data):
        # print("%s: pose\n(%.2f, %.2f,%.2f)\n"%(data.name[-1], data.pose[-1].position.x, data.pose[-1].position.y, data.pose[-1].orientation.z,data.pose[-1].orientation.w))
        print(data)
        return((data.pose[-1].position.x),(data.pose[-1].position.y),(data.pose[-1].orientation.z))
        # print(data.twist[-1])

    def scan(self, data):
        front_min_distance = min(data.ranges[1:10] + data.ranges[-10:])
        left_min_distance = min(data.ranges[11:30])
        right_min_distance = min(data.ranges[-30:-11])
        check_distance = 0.5
        side_check_distance = check_distance - 0.1
        if front_min_distance < check_distance:
            print("front")
            self.move_right()
        else:
            if left_min_distance < right_min_distance and left_min_distance < side_check_distance:
                print("move right")
                self.move_right()
            elif right_min_distance < left_min_distance and right_min_distance < side_check_distance:
                print("move left")
                self.move_left()
            else:
                self.move_forward()
        self.pub.publish(self.rot)
        self.rate.sleep() 


if __name__ == "__main__":
    # env = gym.make("Taxi-v3").env
    # env.reset()
    # # env.render()
    # print(env.action_space)
    # print(env.observation_space)

    # state = env.encode(3, 3, 0, 0) # (taxi row, taxi column, passenger index, destination index)
    # print("State:", state)

    # env.s = state
    # env.render()

    try:
        AvoidObstacle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
