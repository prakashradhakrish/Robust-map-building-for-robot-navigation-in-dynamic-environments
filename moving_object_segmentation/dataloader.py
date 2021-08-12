'''
@author: Prakash Radhakrishnan
@organization: TU Delft
@reference: https://github.com/prakashradhakrish?tab=repositories
@comments: Part of master thesis work
'''


from torchvision.datasets.vision import VisionDataset
import glob
from torchvision import transforms,datasets
from PIL import Image

## need to add input arguments for data loading

class synDataset(VisionDataset):
    def __init__(self, inp_dim):   
        self.transforms = transforms.ToTensor()
        self.inp_dim = inp_dim
        self.rgb_dir = glob.glob("/homebackup/dataset/2011_09_30/Kitti/dataset_odometry_annotation/dataset_for_training/images/*.png")
        self.rgb_dir.sort()
        self.flow_dir = glob.glob("/homebackup/dataset/2011_09_30/Kitti/dataset_odometry_annotation/dataset_for_training/flownet/*.png")
        self.flow_dir.sort()
        self.mask1_dir = glob.glob("/homebackup/dataset/2011_09_30/Kitti/dataset_odometry_annotation/dataset_for_training/mask1/*.png")
        self.mask1_dir.sort()
        self.mask2_dir = glob.glob("/homebackup/dataset/2011_09_30/Kitti/dataset_odometry_annotation/dataset_for_training/mask2/*.png")
        self.mask2_dir.sort()
                  
    def __getitem__(self, index):
        rgb = Image.open(self.rgb_dir[index]).convert('RGB')
        rgb = rgb.resize(self.inp_dim)
        flow = Image.open(self.flow_dir[index]).convert('RGB')
        flow = flow.resize(self.inp_dim)
        mask1 = Image.open(self.mask1_dir[index])
        mask1 = mask1.resize(self.inp_dim)
        mask2 = Image.open(self.mask2_dir[index])
        mask2 = mask2.resize(self.inp_dim)
        
        if self.transforms is not None:
            rgb = self.transforms(rgb)
            flow = self.transforms(flow)
            mask1 = self.transforms(mask1)
            mask2 = self.transforms(mask2)
        return rgb, flow, mask1, mask2

    def __len__(self):  
        return len(self.rgb_dir)

class synDataset_test(VisionDataset):
    def __init__(self, inp_dim):   
        self.transforms = transforms.ToTensor()
        self.inp_dim = inp_dim
        self.rgb_dir = glob.glob("/homebackup/dataset/2011_09_30/Kitti/Thesis_evaluation/05/image_3/*.png")
        self.rgb_dir.sort()
        self.flow_dir = glob.glob("/homebackup/dataset/2011_09_30/Kitti/Thesis_evaluation/05/flownet/*.png")
        self.flow_dir.sort()
        #self.mask_dir = glob.glob("/homebackup/dataset/2011_09_30/downloads/images-20210610T200228Z-001/test/mask/*.png")
        #self.mask_dir.sort()

                  
    def __getitem__(self, index):
        rgb = Image.open(self.rgb_dir[index]).convert('RGB')
        rgb = rgb.resize(self.inp_dim)
        flow = Image.open(self.flow_dir[index]).convert('RGB')
        flow = flow.resize(self.inp_dim)
        #mask = Image.open(self.mask_dir[index])
        #mask = mask.resize(self.inp_dim)

        
        if self.transforms is not None:
            rgb = self.transforms(rgb)
            flow = self.transforms(flow)
            #mask = self.transforms(mask)

        return rgb, flow

    def __len__(self):  
        return len(self.rgb_dir)
    

