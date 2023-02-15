#! /usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
rospy.init_node('rotate_turtlebot')
pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
rate = rospy.Rate(1)
rot = Twist()
rot.angular.z = 0.5
rot.linear.x = 0.5
while not rospy.is_shutdown():
    pub.publish(rot)
    rate.sleep()