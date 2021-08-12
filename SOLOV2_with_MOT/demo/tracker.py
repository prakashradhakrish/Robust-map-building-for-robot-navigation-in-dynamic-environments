'''
Multi object tracking
Author: Prakash Radhakrishnan(PR)
'''
from utils_tracker import iou_mask,mask_2_bbox
import numpy as np
import cv2
import math

def new_tracking(masks,mask_features,object_f1,max_val,classes):

    ## 1. class constraint 2. Dilated Mask for tracking  3. Feature similarity-  added 28/04/21 - PR

    ## iou box
    mask_ious = []
    obj_ref_id = []
    #print(len(masks[0]),len(masks[1]),len(classes[0]),len(classes[1]))
    for i in range(len(masks[1])):    
        # isolating the matched instances in frame 1 for predicted instance 
        # in frame2 based on IOU
        iou_idx_list = []
        m1= (masks[1][i]).astype(float)
        box = mask_2_bbox(m1)
        m1_mod = (m1.copy()*0)
        m1_mod[box[0]:box[2],box[1]:box[3]]=1
        kernel = np.ones((35,35), np.uint8)
        m1_mod = cv2.dilate(m1_mod, kernel, iterations=1)
        for j in range(len(masks[0])):
            if classes[1][i]==classes[0][j]: # adding class based constraint
                m2 = masks[0][j].astype(float)
                iou = iou_mask(m1_mod,m2)
                if iou > 0 :
                    iou_idx_list.append(j)
        mask_ious.append(iou_idx_list)

        ## feature similarity score
        if len(iou_idx_list)>0:
            best_dist = float('inf')
            best_dist_idx = -1 # random value
            for k in iou_idx_list:
                dist = np.linalg.norm(mask_features[0][k]-mask_features[1][i])
                if dist < best_dist:
                    best_dist = dist
                    best_dist_idx = k
                    #print(best_dist)
            if best_dist_idx == -1 or object_f1[best_dist_idx] in obj_ref_id:
                obj_ref_id.append(max(max_val,max(obj_ref_id))+1)
            else:
                obj_ref_id.append(object_f1[best_dist_idx])
        else:
            if len(obj_ref_id) > 0:
                obj_ref_id.append(max(max_val,max(obj_ref_id))+1)
            else:
                obj_ref_id.append(max_val+1)
            
    return obj_ref_id