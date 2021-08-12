'''
Multi object tracking
Author: Prakash Radhakrishnan(PR)
'''

import numpy as np
import torch
import os

# feature embedder model for extracting feature vectors
def predictor(img,predictions,interested_labels,feature_extractor):
    pred = (predictions['instances'].pred_masks).cpu()
    cls_labels = (predictions['instances'].pred_classes).cpu()
    mask = []
    res_feat = []
    classpred = []
    for i in range(len(pred)):
        if cls_labels[i] in interested_labels:
            pred1 = np.asarray(pred[i]).astype("uint8")
            mask.append(pred1)
            img1= img.copy()
            img1[pred1 == 0] = 0
            img1 = torch.tensor(img1,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
            out_feat = feature_extractor(img1).view(-1)
            res_feat.append(out_feat.detach().numpy())
            classpred.append(cls_labels[i].detach().numpy())
    return mask, res_feat,classpred