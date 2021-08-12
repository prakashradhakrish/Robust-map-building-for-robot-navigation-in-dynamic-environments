## code for subscribing the rosmsg from the bag to folder
## Author: Prakash Radhakrishnan

'''
Program coded to subscribe rosmsg to the image folders

Input: Rostopic for any image
Output: image saved in a folder
'''

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    parser = argparse.ArgumentParser(description="Extract image frame based on topic from ROS bag.")
    parser.add_argument("bag_dir", help="Directory for ROS bag.")
    parser.add_argument("image_dir", help="Folder of output directory to save image")
    parser.add_argument("ros_image_topic", help="topic to subscribe")

    args = parser.parse_args()


    bag = rosbag.Bag(args.bag_dir, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.ros_image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(os.path.join(args.image_dir, "frame%06i.png" % count), cv_img)
        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()