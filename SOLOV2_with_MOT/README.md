# SOLOV2 with multi-object tracking

Repository contains code implemented as part of final Master thesis.

SOLOV2 code is adapted from AdelaiDet written on detectron2 as PyTorch framework.

## Pre-requisites
- For installation check the repo - https://github.com/aim-uofa/AdelaiDet.git
- Make sure the weights 'SOLOv2_R50_3x.pth' is downloaded

## Geometric approach using epipolar geometry
- 'dynamic_point_detect_new.py' - The implementation include Fundamental matrix calculation, lucas-Kanade optical flow vector, epipolar line determination.

## Multi-object tracking
- 'tracker.py' uses track by detection which follows Seperation detection and embedding 
- In addition to the feature similarity based on the euclidean distance, Class constraint and IOU filtering is used
- Feature embedder model trail done and Resnet18 is chosen

## RUN the code
1. To save  mask and track image output as folder
  ```
  python demo/solov2node.py --config-file configs/SOLOv2/R50_3x.yaml --confidence-threshold=0.3 --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth
  ```
2. To publish the mask image as '/mask_image' and track image as '/track_image' by subscribing rgb input as '/camera/usb_cam_1/image_raw'
  ```
  python demo/solov2node_ros.py --config-file configs/SOLOv2/R50_3x.yaml --confidence-threshold=0.3 --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth
  ```
## Output


## Acknowledgement

- Wang, X., Zhang, R., Kong, T., Li, L., & Shen, C. (2020). **SOLOv2: Dynamic and fast instance segmentation**. arXiv preprint arXiv:2003.10152. **[PDF](https://arxiv.org/pdf/2003.10152.pdf)**
