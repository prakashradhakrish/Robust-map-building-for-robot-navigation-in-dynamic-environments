#!/usr/bin/env python

'''
Instance segmentation as ROS package

SOLOV2 instance segmenatation based on https://github.com/aim-uofa/AdelaiDet.git 

credits to AdelaiDet: A Toolbox for Instance-level Recognition Tasks

Input arguments: RGB image subscribed as rostopic
Output : Track and instance mask published as rostopic

User input: Directories for RGB, mask, track image, interested label

Author: Prakash Radhakrishnan(PR)
'''

import os
import threading
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
from tqdm import tqdm
import torch

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor
import torchvision.models as models

from adet.config import get_cfg

import numpy as np
from cv_bridge import CvBridge
import rospy
import visualiser
import post_processing_model
from tracker import new_tracking

try: 
    import queue
except ImportError:
    import Queue as queue

from sensor_msgs.msg import Image

class Solov2node(object):
    def __init__(self,cfg):
        self._cv_bridge = CvBridge()
        self.cfg = cfg

        self._visualization = rospy.get_param('~visualization', True)

        # Create model object in inference mode.
        self.model_predictor = DefaultPredictor(self.cfg)

        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._publish_rate = rospy.get_param('~publish_rate', 100)
    
    def order_image_files(self,directory):
        os.chdir(directory)
        images_directory = glob.glob("*.png") # user input of corresponding file
        images_directory.sort()
        return images_directory

    def run(self):
        # Ros param
        mask_pub = rospy.Publisher('/mask_image', Image, queue_size=1)
        track_pub = rospy.Publisher('/track_image', Image, queue_size=1)
        sub = rospy.Subscriber('/camera/usb_cam_1/image_raw', Image,self._image_callback, queue_size=1)
        rate = rospy.Rate(self._publish_rate) 

        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue 

            if msg is not None:
                img = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

                # Model prediciton using default detectron2 library
                predictions,fpn_feat,mask_feature = self.model_predictor(img)

                interested_labels = [0,1,2] # User input needed
                pred = (predictions['instances'].pred_masks).cpu()
                cls_labels = (predictions['instances'].pred_classes).cpu()
                mask = []
                for i in range(len(pred)):
                    if cls_labels[i] in interested_labels:
                        pred1 = np.asarray(pred[i]).astype("uint8")
                        mask.append(pred1)


                # mask Generation
                mask_img = np.array(img[:, :, ::-1])
                mask_img = np.zeros((mask_img.shape[0],mask_img.shape[1]))
                interested_labels = [0,1,2]
                mask_img = visualiser.visualise_mask_instance_custom(mask_img,mask)

                mask_image = np.array(mask_img, dtype = np.uint8)

                #Ros header files definition
                Stamp = rospy.Time.now()
                #mask
                mask_msg = self._cv_bridge.cv2_to_imgmsg(mask_image, encoding="mono8")
                mask_msg.header.stamp = Stamp    
                #publish
                mask_pub.publish(mask_msg)
                track_pub.publish(mask_msg)
            rate.sleep()

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.SOLOV2.SCORE_THR = args.confidence_threshold
    cfg.MODEL.SOLOV2.UPDATE_THR = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def main():
    rospy.init_node('Solov2+tracking_node')
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    node = Solov2node(cfg)
    node.run()

if __name__ == '__main__':
    main()
