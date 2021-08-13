# Moving object segmentation

Repository contains code implemented as part of final Master thesis.

This code is implemented from Scratch on PyTorch framework.

## Pre-requisites
- pyTorch 1.7.1
- If optical flow image needs to be generated from the scratch make sure to install dependency from FlowNetV2 (https://github.com/NVIDIA/flownet2-pytorch.git)

## Post processing
- Current inference does not produce accurate segmentation contour for moving object. So the result is combined with SOLOV2 to achieve perfect contour which could be used in SLAM.


## RUN the code
1. Jupyter Notebook- Recommended to use Two_stream_motion_segmentation.ipynb for running the code 
2. If the code has to be run on server run below code. Make sure to check the configuration of the model, weights and directory of image folders used.
```
python main.py
```
Todo: Get the configuration, weight and directory of image folder as arguments
## Output


## Acknowledgement

- Siam, M., Mahgoub, H., Zahran, M., Yogamani, S., Jagersand, M., & El-Sallab, A. (2018, November). **Modnet: Motion and appearance based moving object detection network for autonomous driving**. In 2018 21st International Conference on Intelligent Transportation Systems (ITSC) (pp. 2859-2864). IEEE. **[PDF](https://arxiv.org/pdf/1709.04821.pdf)**
