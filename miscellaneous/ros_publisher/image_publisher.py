#!/usr/bin/env python

## code for publishing the images in folder as ros msg. 
## Author: Prakash Radhakrishnan
## Date: 10/02/2021

'''
Program coded to publish rosmsg from the image folders

Input: images,mask,track,depth(if needed) directory
Output: Publishes Rostopics
'''

# Default libraries for python
import glob
import cv2
import numpy as np
import math
import os
import time

# Ros libraries
import sensor_msgs.msg
from sensor_msgs.msg import Temperature
from std_msgs.msg import Bool, Int8
import rospy
from cv_bridge import CvBridge


# Store the directory list of images 
def order_image_files(directory):
  os.chdir(directory)
  images_directory = glob.glob("*.png") # user input of corresponding file
  images_directory.sort()
  return images_directory

# Function which publishes the image in sequence based on mask_rcnn_status
def image_reader(image_directory,mask_directory,track_directory,dynamic_text,rate_publish,depth_enabled,depth_directory=None,association_text=None):
  bridge = CvBridge()
  if (depth_enabled):
     file = open(association_text,"r")
     file.seek(0)
     data_list = file.readlines()
     file.close()

  file1 = open(dynamic_text,"r")
  file1.seek(0)
  dyn_list = file1.readlines()
  file1.close()

  dyn_img_list = []
  dyn_count_list = []
  for dyn_file in dyn_list:
      split_corres = str(dyn_file).split()
      dyn_img_list.append(split_corres[0])
      dyn_count_list.append(split_corres[1])

  file1 = open("/home/prakash/thesis_result/ORB_SLAM2/kitti_04/groundtruth/times.txt","r")
  file1.seek(0)
  gt_list = file1.readlines()
  file1.close()

  time_list = []
  for dyn_file in gt_list:
      split_corres = str(dyn_file).split()
      time_list.append(split_corres[0])

  dyn_list = []
  if (depth_enabled):
    image_list = []
    mask_list = []
    depth_list = []
    track_list = []
    for associate_file in data_list:
      split_corres = str(associate_file).split()
      image_list.append(os.path.join(image_directory,(split_corres[0]+".png")))
      mask_list.append(os.path.join(mask_directory,(split_corres[0]+"_mask.png")))
      track_list.append(os.path.join(track_directory,(split_corres[0]+"_track.png")))
      depth_list.append(os.path.join(depth_directory,(split_corres[2]+".png")))
      idx = dyn_img_list.index(split_corres[0])
      dyn_list.append(int(dyn_count_list[idx]))
  else:
      image_list = order_image_files(image_directory)
      mask_list = order_image_files(mask_directory)
      track_list = order_image_files(track_directory)
      for img in image_list:
        idx = dyn_img_list.index(os.path.splitext(os.path.basename(img))[0])
        dyn_list.append(int(dyn_count_list[idx]))
        
    

  rospy.loginfo('Number of images found: ' + str(len(image_list)))
  rospy.loginfo('Number of mask found: ' + str(len(mask_list)))
  if (depth_enabled):
     rospy.loginfo('Number of depth found: ' + str(len(depth_list)))
  
  # publisher initialisation
  rate = rospy.Rate(rate_publish)
  image_publisher = rospy.Publisher('/camera/usb_cam_1/image_raw', sensor_msgs.msg.Image, queue_size = 1)
  mask_publisher = rospy.Publisher('/mask_image', sensor_msgs.msg.Image, queue_size = 1)
  track_publisher = rospy.Publisher('/track_image', sensor_msgs.msg.Image, queue_size = 1)
  depth_publisher = rospy.Publisher('/camera/depth/image', sensor_msgs.msg.Image, queue_size = 1)
  dynamic_publisher = rospy.Publisher('/dynamic_count', sensor_msgs.msg.Image, queue_size = 1)
    #print('out '+str(status_check_for_pub))
  for i in range(len(image_list)):
    if rospy.is_shutdown():
      break
    #Stamp = rospy.Time.now()
    Stamp = rospy.Time.from_sec(float(os.path.splitext(os.path.basename(image_list[i]))[0]))
    #Stamp = rospy.Time.from_sec(float(time_list[i]))
    #print(type(Stamp))
    image = cv2.imread(os.path.join(image_directory,image_list[i]))
    image_msg = bridge.cv2_to_imgmsg(np.asarray(image[:,:]), encoding='bgr8')
    image_msg.header.frame_id = os.path.splitext(os.path.basename(image_list[i]))[0]
    image_msg.header.stamp = Stamp

    mask = cv2.imread(os.path.join(mask_directory,mask_list[i]),0)
    mask[mask==1]=255
    mask_msg = bridge.cv2_to_imgmsg(np.asarray(mask[:,:]), encoding='mono8')
    mask_msg.header.frame_id = os.path.splitext(os.path.basename(mask_list[i]))[0]
    mask_msg.header.stamp = Stamp

    track = cv2.imread(os.path.join(track_directory,track_list[i]),0)
    track_msg = bridge.cv2_to_imgmsg(np.asarray(track[:,:]), encoding='mono8')
    track_msg.header.frame_id = os.path.splitext(os.path.basename(track_list[i]))[0]
    track_msg.header.stamp = Stamp

    dynamic_count_img = (np.ones((3,3), dtype=np.uint8))*dyn_list[i]
    dynamic_count_msg = bridge.cv2_to_imgmsg(np.asarray(dynamic_count_img[:,:]), encoding='mono8')
    dynamic_count_msg.header.frame_id = (os.path.splitext(os.path.basename(image_list[i]))[0])
    dynamic_count_msg.header.stamp = Stamp

    if (depth_enabled):
       depth = cv2.imread(os.path.join(depth_directory,depth_list[i]),0)
       depth_msg = bridge.cv2_to_imgmsg(np.asarray(depth[:,:]), encoding='mono8')
       depth_msg.header.frame_id = (os.path.splitext(os.path.basename(depth_list[i]))[0])
       depth_msg.header.stamp = Stamp
    
    image_publisher.publish(image_msg)
    mask_publisher.publish(mask_msg)
    track_publisher.publish(track_msg)
    if (depth_enabled):
       depth_publisher.publish(depth_msg)
    dynamic_publisher.publish(dynamic_count_msg)

    rate.sleep()
  rospy.loginfo('Publishing of images finished!!!')

if __name__ == "__main__":
  rospy.init_node('image_publisher')
  image_directory = "/homebackup/bagfiles/bagfiles/final_bagfiles/realtime/image_folder/"
  depth_directory = "/home/prakash/git/ORB_SLAM2/dataset/rgbd_dataset_freiburg3_walking_static/depth/"
  mask_directory = "/homebackup/bagfiles/bagfiles/final_bagfiles/realtime/mask/"
  track_directory = "/homebackup/bagfiles/bagfiles/final_bagfiles/realtime/track/"
  association_text = "/home/prakash/git/ORB_SLAM2/associate.txt"
  dynamic_text = "/homebackup/bagfiles/bagfiles/final_bagfiles/realtime/realtime_chair.txt"

  image_reader(image_directory,mask_directory,track_directory,dynamic_text,10,False)


