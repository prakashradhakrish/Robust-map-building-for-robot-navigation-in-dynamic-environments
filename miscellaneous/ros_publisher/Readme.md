Files description

1. image_publisher.py 
   Publishes following output rostopic
  - /camera/usb_cam_1/image_raw - input image
  - /mask_image - segmentation image
  - /track_image - Tracking instance image
  - /camera/depth/image - Depth image
  - /dynamic_count - Number of dynamic objects in the scene
  Execution of the code as follows
  ```
  python mask_rcnn_node.py
  '''
  To save the the published topics into rosbag, run following code
  '''
  rosbag record -a
  '''
2. bagtoimageframe.py - Convert bag file to images in a folder 
  Execution of the code as follows
  ```
  python bagtoimageframe.py ./xx.bag ./directory /rostopic
  '''
