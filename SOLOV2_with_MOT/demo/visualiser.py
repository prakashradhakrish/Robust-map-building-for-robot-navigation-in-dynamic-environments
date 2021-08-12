# Created: march 29 2021, PR~
import os
import cv2
import numpy as np
import dynamic_point_detect_new

def visualise_mask_instance(mask_image,mask,corres_obj_id):
    '''
    Input args: 
    Mask_image -> white image with dimension HxWx3
    Predictions -> Dict of Tensor values received directly from solov2 model
    Interested_labels -> list of class labels number used for mask

    Output:
    mask_image -> black mask with white background with dimension HxWx3        
    '''
    dyn_count = 0
    track_img = np.zeros((mask_image.shape[0],mask_image.shape[1]))
    #print(corres_obj_id)
    for j in range(len(mask)):
        pred1 = np.asarray(mask[j]).astype("uint8")
        track_id = corres_obj_id[j] 
        dyn_count +=1
        x,y = np.where(pred1==1)
        for i in range(len(y)):
            if track_img[x[i],y[i]]==0:
                track_img [x[i],y[i]] = j+1
            if mask_image[x[i],y[i]]==255:
                mask_image[x[i],y[i]] = j+1
    return mask_image,dyn_count,track_img

def visualise_mask_instance_custom(mask_image,mask):
    '''
    Input args: 
    Mask_image -> white image with dimension HxWx3
    Predictions -> Dict of Tensor values received directly from solov2 model
    Interested_labels -> list of class labels number used for mask

    Output:
    mask_image -> black mask with white background with dimension HxWx3        
    '''
    
    for j in range(len(mask)):
        pred1 = np.asarray(mask[j]).astype("uint8")
        x,y = np.where(pred1==1)
        for i in range(len(y)):
            if mask_image[x[i],y[i]]==0:
                mask_image [x[i],y[i]] = 255
    return mask_image

def visualise_mask(mask_image,mask,corres_obj_id):
    '''
    Input args: 
    Mask_image -> white image with dimension HxWx3
    Predictions -> Dict of Tensor values received directly from solov2 model
    Interested_labels -> list of class labels number used for mask

    Output:
    mask_image -> black mask with white background with dimension HxWx3        
    '''
    dyn_count = 0
    track_img = np.zeros((mask_image.shape[0],mask_image.shape[1]))
    #print(corres_obj_id)
    for j in range(len(mask)):
        pred1 = np.asarray(mask[j]).astype("uint8")
        track_id = corres_obj_id[j] 
        dyn_count +=1
        x,y = np.where(pred1==1)
        for i in range(len(y)):
            if track_img[x[i],y[i]]==0:
                track_img [x[i],y[i]] = track_id
            if mask_image[x[i],y[i]]==255:
                mask_image[x[i],y[i]] = 255 -200
    return mask_image,dyn_count,track_img

def mask_optcl_points(mask_image_dyn, mask_ind, points,prediction_counter):
    x,y = np.where(mask_ind==1)
    mask_points = []   
    for i in range(len(x)):
        mask_points.append([y[i],x[i]])

    flow_counter = 0
    if len(points) > 0:
        for pt in points:
            pt = list(pt)
            if pt in mask_points:
                flow_counter = flow_counter + 1
    else:
        #print("No enough dynamic points")
        flow_counter = 0

    #print("flow_counter",flow_counter)
    counter_dyn_flag =  0
    if flow_counter > 3 : # can be changed manually 
        for i in range(len(y)):
            if mask_image_dyn[x[i],y[i]] == 255:
                #mask_image_dyn[x[i],y[i]] = 255 -(prediction_counter+100)
                mask_image_dyn[x[i],y[i]] = 255 -100
            #if (255 -(prediction_counter+100)) <= 55:
                #print("Dynamic mask assumption of exceeded")
        counter_dyn_flag =  1

    else:
        for i in range(len(y)): 
            if mask_image_dyn[x[i],y[i]] == 255:
                #mask_image_dyn[x[i],y[i]] = 255 -(prediction_counter+200)
                mask_image_dyn[x[i],y[i]] = 255 - 200
            #if (255 -(prediction_counter+100)) == 0:
                #print("Mask count of 55 exceeded")
                                    
        
    return mask_image_dyn,counter_dyn_flag


def visualise_mask_dynamic(prev_image,image,prev_mask,mask_image,mask,obj_ref,prev_track_img):
    '''
    Input args: 
    Mask_image -> white image with dimension HxWx3
    Predictions -> Dict of Tensor values received directly from solov2 model
    Interested_labels -> list of class labels number used for mask
    prev_mask -> prev_mask with dimension HxWx3 needed for optical flow

    Output:
    mask_image -> mask with two distinction (static, dynamic) with white background with dimension HxWx3        
    '''
    track_img = np.zeros((mask_image.shape[0],mask_image.shape[1]))
    #print(obj_ref)
    for i in range(len(mask)):
        pred1 = np.asarray(mask[i]).astype("uint8")
        mask_image [:, :] = np.where(pred1 == 1, mask_image [:, :]*0, mask_image [:, :])
        track_id = obj_ref[i]
        if track_id>0:
            x,y = np.where(pred1==1)
            for i in range(len(y)):
                if track_img[x[i],y[i]]==0:
                    track_img [x[i],y[i]] = track_id

    points = dynamic_point_detect_new.detect_moving_mask_new(prev_image,image,prev_mask,mask_image,prev_track_img,track_img)

    dyn_count = 0
    prediction_counter = 0
    mask_image_dyn = np.ones((mask_image.shape[0],mask_image.shape[1]))*255
    for i in range(len(mask)):
        mask_ind = np.asarray(mask[i]).astype("uint8")
        mask_image_dyn,counter_dyn_flag = mask_optcl_points(mask_image_dyn, mask_ind, points,prediction_counter)
        dyn_count = dyn_count + counter_dyn_flag
        prediction_counter+=1
    
    return mask_image_dyn,dyn_count,track_img
