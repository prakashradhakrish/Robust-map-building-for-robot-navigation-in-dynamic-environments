# ORB-SLAM2 with dynamic object removal and Semantic segmentation

Repository contains code implemented as part of final Master thesis.

The code of monocular camera as a ROS pipeline is modified to handle dynamic objects and build semantic map.


## Pre-requisites
- Follow instruction of ORB-SLAM2 prerequisites based on (https://github.com/raulmur/ORB_SLAM2)
- Follow instruction of PCL library for saving map based on (https://github.com/PointCloudLibrary/pcl.git)
- Follow instruction for installation of Octomap based on (https://github.com/OctoMap/octomap.git). This is optional needed for map visualisation.

## Build
- Clone the repository:
```
git clone https://github.com/prakashradhakrish/Robust-map-building-for-robot-navigation-in-dynamic-environments/ORB_SLAM2_dynamic_semmap.git ORB_SLAM2_dynamic_semmap
```

We adapted same script from ORB-SLAM2 `build.sh`.
```
cd ORB_SLAM2_dynamic_semmap
chmod +x build.sh
./build.sh
```
- build ROS package
1. Add the path including *Examples/ROS/ORB_SLAM2* to the ROS_PACKAGE_PATH environment variable. Open .bashrc file and add at the end the following line. Replace PATH by the folder where you cloned ORB_SLAM2_dynamic_semmap:
  ```
  export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/ORB_SLAM2/Examples/ROS
  ```
  
 We adapted same script from ORB-SLAM2 `build_ros.sh`.

  ```
  chmod +x build_ros.sh
  ./build_ros.sh
  ```
2. For Running Monocular Node, we need a monocular input from topic `/camera/usb_cam_1/image_raw, a segmentation mask topic '/mask_image', a tracking instance image topic '/track_image' and a dynamic count '/dynamic_count'. Replace xx with recommeded configuration
  ```
  rosrun ORB_SLAM2 Mono Vocabulary/ORBvoc.txt Examples/Monocular/xx.yaml True
  ```
Todo: Dynamic count topic need to be removed

## Output
Click on the image to play the video
[![ORB_SLAM2 fused with deep learning models](https://img.youtube.com/vi/XMLr5PRF1vA/0.jpg)](https://www.youtube.com/watch?v=XMLr5PRF1vA&ab_channel=PrakashR)

## Acknowledgement

- Raúl Mur-Artal and Juan D. Tardós. **ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras**. *IEEE Transactions on Robotics,* vol. 33, no. 5, pp. 1255-1262, 2017. **[PDF](https://128.84.21.199/pdf/1610.06475.pdf)**
- Bescos, B., Fácil, J. M., Civera, J., & Neira, J. (2018). **DynaSLAM: Tracking, mapping, and inpainting in dynamic scenes**. IEEE Robotics and Automation Letters, 3(4), 4076-4083. **[PDF](https://arxiv.org/pdf/1806.05620.pdf)**
