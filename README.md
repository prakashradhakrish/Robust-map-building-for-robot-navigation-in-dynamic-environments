# Robust-map-building-for-robot-navigation-in-dynamic-environments

Repository contains code implemented as part of final Master thesis

### List of folders and details

- [**ORB_SLAM2_dynamic_semmap**](https://github.com/prakashradhakrish/Robust-map-building-for-robot-navigation-in-dynamic-environments/tree/main/ORB_SLAM2_dynamic_semmap) - ORB-SLAM2 base implementation is modified to accept the segmentation input and tracking instance input to handle dynamic objects during tracking and perform semantic mapping
- [**SOLOV2_with_MOT**](https://github.com/prakashradhakrish/Robust-map-building-for-robot-navigation-in-dynamic-environments/tree/main/MaskRCNN_ros) - SOLOV2 base implementation is adapted to run as ROS package. SOLOV2 output is processed with epipolar constraint and multi-object tracking to generate segmentation mask with dynamic details and track image with instance id.
- [**moving_object_segmentation**] (https://github.com/prakashradhakrish/Robust-map-building-for-robot-navigation-in-dynamic-environments/tree/main/moving_object_segmentation) - Two stream encoder-decoder architecture based on VGG and FCN network that accepts the RGB image and optical flow image to output spatial semantic segmentation and moving object segmentation
- [**MaskRCNN_ros**] (https://github.com/prakashradhakrish/Robust-map-building-for-robot-navigation-in-dynamic-environments/tree/main/MaskRCNN_ros) - MaskRCNN base implementation is adapted to run as ROS package. This is implemented for comparing the benefits of single stage instance segmentation(SOLOV2) against two stage instance segmentation(MaskRCNN)
- [**miscellaneous**] (https://github.com/prakashradhakrish/Robust-map-building-for-robot-navigation-in-dynamic-environments/tree/main/miscellaneous) - The folder contains various metric used for evaluating segmentation, tracking and odometry. It also contain some useful scripts for publishing ros message

