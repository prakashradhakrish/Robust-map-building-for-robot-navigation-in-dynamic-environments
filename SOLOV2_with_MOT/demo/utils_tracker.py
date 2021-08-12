import numpy as np
import os
import glob
import cv2

"""
Util functions for image tracking
"""
# Ordering the image list in a given directory with extension PNG
def order_image_files(directory):
    os.chdir(directory)
    images_directory = glob.glob("*.png") # user input of corresponding file
    images_directory.sort()
    return images_directory

# convert segmentation output to bounding box 
def mask_2_bbox(mask_inp): # input takes numpy needs to be detached in cpu
    idx = np.argwhere(mask_inp)
    (y1, x1), (y2, x2) = idx.min(0), idx.max(0) 
    bbox = np.asarray([y1,x1,y2,x2], dtype = np.uint32)
    return bbox

# find template matches for given image and template using opencv libraries
def template_matching(img_t,template):
    w, h = template[:,:,0].shape[::-1]
    method = eval('cv2.TM_CCOEFF_NORMED') # needs to experiment further to freeze
    res = cv2.matchTemplate(img_t,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)   
    template_bbox = np.asarray([top_left[1],top_left[0],bottom_right[1],bottom_right[0]],dtype = np.uint32)   
    return template_bbox

# compute iou metrics for given two segmenation masks
def iou_mask(prediction,ground_truth):
    intersection = np.logical_and(prediction, ground_truth)
    union = np.logical_or(prediction, ground_truth)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# to compute iou of bbox we use pytorch inbuilt function bops.box_iou

def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list
