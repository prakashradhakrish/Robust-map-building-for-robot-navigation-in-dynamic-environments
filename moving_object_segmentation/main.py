#!/usr/bin/env python
'''
@author: Prakash Radhakrishnan
@organization: TU Delft
@reference: https://github.com/prakashradhakrish?tab=repositories
@comments: Part of master thesis work
'''

import numpy as np
from datetime import datetime
import time
import rospy
from model_vgg import modnet,modnet_2
import torch
import glob,os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import cv2



class modnet_vgg(object):
    def __init__(self):
        self.mode = 'inference'
        self.model = modnet()
        #loading weights
        self.model.load_state_dict(torch.load('/home/prakash/trial_scripts/cpkt_modnet_2021_07_18_184304_30ep_cfg1_closs_actdata_orgdim.pth'))
        if self.mode =='inference':
            self.model = self.model.eval()
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.model = self.model.cuda()
        self.inp_dim = (1226,370)
        self.transforms = transforms.ToTensor()
    
    def order_image_files(self,directory):
        os.chdir(directory)
        images_directory = glob.glob("*.png") # user input of corresponding file
        images_directory.sort()
        return images_directory
    
    def iou_calc(self,target,prediction,i):
        temp_target = np.zeros(target.shape)
        temp_target[target==i]=1
        intersection = np.logical_and(temp_target, prediction)
        union = np.logical_or(temp_target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def run(self):
        while not rospy.is_shutdown():
            input_directory = "/homebackup/dataset/2011_09_30/downloads/images-20210610T200228Z-001/test/images/"
            image_list = self.order_image_files(input_directory)
            flow_directory =  "/homebackup/dataset/2011_09_30/downloads/images-20210610T200228Z-001/test/flownet/"
            #solov2_directory = "/homebackup/dataset/2011_09_30/Kitti/Thesis_evaluation/05/track/"
            

            start_time = time.time()
            for singleimage_dir in tqdm(image_list):
                # Reading Images and assigning names to mask and image
                image_name = os.path.splitext(os.path.basename(singleimage_dir))[0]
                rgb_path = os.path.join(input_directory,singleimage_dir)
                flow_path = os.path.join(flow_directory,image_name+'-vis.png')
                #solo_track_path = os.path.join(solov2_directory,image_name+'_track.png')
                mask_name = image_name+"_motion"+".png"
                motion_solov2 = "/homebackup/dataset/2011_09_30/downloads/images-20210610T200228Z-001/test/predicted/" +  mask_name


                inputimage = Image.open(rgb_path).convert('RGB')
                orig_size = inputimage.size
                inputimage = inputimage.resize(self.inp_dim)
                inputimage = self.transforms(inputimage)

                flowimage = Image.open(flow_path).convert('RGB')
                flowimage = flowimage.resize(self.inp_dim)
                flowimage = self.transforms(flowimage)

                if self.gpu_available:
                    inputimage = inputimage.cuda()
                    flowimage = flowimage.cuda()
                    
                outputs = self.model(inputimage.unsqueeze(0),flowimage.unsqueeze(0))
                pred = torch.sigmoid(outputs[1])
                pred[pred<0.6]=0
                pred = torch.argmax(pred, dim=1)

                # Code to include mask only if it matches the flow is visible
                #flow = np.ones((flowimage.detach().cpu().permute(1,2,0).numpy()).shape)
                #flow = flowimage.detach().cpu().permute(1,2,0).numpy()
                #flow[(flowimage.detach().cpu().permute(1,2,0).numpy())>0.98]=0
                #flow = np.mean((flow),axis=2)
                #flow[flow>0]=1
                #prediction = pred.detach().cpu().numpy().squeeze(0)*flow
                
                prediction = pred.detach().cpu().numpy().squeeze(0)
                prediction = np.array(prediction, dtype='uint8')
                prediction = cv2.resize(prediction, (orig_size), interpolation = cv2.INTER_AREA)


                # code to include solov2 ouput with motion segmentation network
                '''
                solo_mask = cv2.imread(solo_track_path,0)
                mask = np.ones(solo_mask.shape)
                for i in list(np.unique(solo_mask)):
                    if i !=0:
                        if i in list(np.unique(prediction*solo_mask)):
                            iou = self.iou_calc(solo_mask,prediction,i)
                            if iou>0.1:
                                mask[solo_mask==i]=155
                            else:
                                mask[solo_mask==i]=55
                        else:
                            mask[solo_mask==i]=55
                            
                mask[mask==0]=255
                '''
                final_mask = Image.fromarray(prediction.astype('uint8'))
                final_mask.save(motion_solov2)
                

            print("--- %s seconds ---" % (time.time() - start_time))
            break

def main():
    rospy.init_node('modnet_vgg')
    node = modnet_vgg()
    node.run()

if __name__=="__main__":
    main()