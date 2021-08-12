#!/usr/bin/env python

'''
Base code is coupled as ROS package

SOLOV2 instance segmenatation based on https://github.com/aim-uofa/AdelaiDet.git 

credits to AdelaiDet: A Toolbox for Instance-level Recognition Tasks

Input arguments: RGB image from the folder
Output : Track and instance mask saved into seperate folder

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
        text_file_name = "realtime_monitor.txt"
        txt_file = open(text_file_name,"w")   

        while not rospy.is_shutdown():  
            input_directory = "/homebackup/bagfiles/bagfiles/final_bagfiles/realtime/image_folder/"          
            image_list = self.order_image_files(input_directory)
            counter = 0
            start_time = time.time()
            resnet18 = models.resnet18(pretrained=True)
            feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:8])
            overall_tracker = []
            max_val = 0
            corres_obj_id = []
            corres_obj_id_len = []
            masks = []
            features = []
            classes = []
            for singleimage_dir in tqdm(image_list):
                # Reading Images and assigning names to mask and image
                image_name = os.path.splitext(os.path.basename(singleimage_dir))[0]
                path = os.path.join(input_directory,singleimage_dir)
                mask_name = image_name+"_mask"+".png"
                track_name = image_name+"_track"+".png"
                mask_path = "/homebackup/bagfiles/bagfiles/final_bagfiles/realtime/mask_monitor/" +  mask_name
                track_path = "/homebackup/bagfiles/bagfiles/final_bagfiles/realtime/track_monitor/" +  track_name 
                img = read_image(path, format="BGR")

                # Model prediciton using default detectron2 library
                predictions,fpn_feat,mask_feature = self.model_predictor(img)

                # mask Generation
                mask_img = np.array(img[:, :, ::-1])
                mask_img = np.ones((mask_img.shape[0],mask_img.shape[1]))*255
                #interested_labels = [0,1,2,5,7]    
                #interested_labels = [0,1,2]
                interested_labels = [62]
                mask, res_feat, classpred = post_processing_model.predictor(img,predictions,interested_labels,feature_extractor)   
                if counter == 0: # change to 0
                    object_id = []
                    for i in range(len(mask)):
                        object_id.append(i+1)
                    #print(object_id)
                    corres_obj_id.append(object_id)
                    corres_obj_id_len.append(len(object_id))
                    overall_tracker.append(object_id)
                    masks.append(mask)
                    classes.append(classpred)
                    features.append(res_feat)
                    max_val = len(mask)-1
                    mask_img,dyn_counter,track_img = visualiser.visualise_mask(mask_img,mask,corres_obj_id[-1])
                
                    prev_image = img
                    prev_mask = mask_img
                    prev_track_img = track_img
                    txt_file.write(image_name+" "+str(dyn_counter)+" \n")
                else:
                    if len(classpred) == 0:
                        overall_tracker.append([])
                        corres_obj_id.append(0)
                        track_img = mask_img.copy()*0
                        txt_file.write(image_name+" "+"0"+" \n")
                    else:                
                        masks.append(mask)
                        classes.append(classpred)
                        features.append(res_feat)
                        nonzeroind = np.nonzero(corres_obj_id_len)[0]
                        closest_idx = np.max(nonzeroind)
                        temp_mask=[masks[closest_idx] , masks[-1]]
                        temp_feat=[features[closest_idx] , features[-1]]
                        temp_class=[classes[closest_idx] , classes[-1]]
                        obj_corresp_frame2 = new_tracking(temp_mask,temp_feat,corres_obj_id[closest_idx],max_val,temp_class)
                        corres_obj_id.append(obj_corresp_frame2)    
                        corres_obj_id_len.append(len(obj_corresp_frame2))
                        overall_tracker.append(obj_corresp_frame2)
                        max_val = max(max_val,max(obj_corresp_frame2))
                        mask_img,dyn_counter,track_img = visualiser.visualise_mask_dynamic(prev_image,img,prev_mask,mask_img,mask,obj_corresp_frame2,prev_track_img)
                        prev_image = img
                        prev_mask = mask_img
                        prev_track_img = track_img
                        masks = masks[-1:]
                        features = features[-1:]
                        classes = classes [-1:]
                        corres_obj_id = corres_obj_id [-1:]
                        corres_obj_id_len = corres_obj_id_len [-1:]
                        txt_file.write(image_name+" "+str(dyn_counter)+" \n")

                mask_image = np.array(mask_img, dtype = np.uint8)
                track_img = np.array(track_img, dtype = np.uint8)   
                cv2.imwrite(mask_path,mask_image)
                cv2.imwrite(track_path,track_img)

                counter+=1
            print("Total_time_of_execution",start_time- time.time())
            txt_file.close()
            break

            rate.sleep()

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
    rospy.init_node('Solov2node')
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    node = Solov2node(cfg)
    node.run()

if __name__ == '__main__':
    main()
