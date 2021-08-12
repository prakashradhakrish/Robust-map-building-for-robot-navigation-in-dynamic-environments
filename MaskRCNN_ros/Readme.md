# MaskRCNN ROS 

Repository contains code implemented as part of final Master thesis.

The part of MaskRCNN code is adapted from matterport(https://github.com/matterport/Mask_RCNN.git)

## Pre-requisites
- Ensure the mask_rcnn_coco.h5 is downloaded in ROS home path

## ROS topic lost
- Input image subscribed from **/camera/usb_cam_1/image_raw**
- Mask image published as **/mask_rcnn/mask**
## RUN the code
  ```
  python mask_rcnn_node.py
  ```

## Output


## Acknowledgement

- He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). **Mask r-cnn**. In Proceedings of the IEEE international conference on computer vision (pp. 2961-2969).. **[PDF](https://arxiv.org/pdf/1703.06870.pdf)**
