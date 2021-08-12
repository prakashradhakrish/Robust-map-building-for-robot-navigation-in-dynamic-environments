'''
@author: Prakash Radhakrishnan
@organization: TU Delft
@reference: https://github.com/prakashradhakrish?tab=repositories
@comments: Part of master thesis work
'''

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

# Base Encoder 
class encoder(nn.Module):
    '''
    Vgg16 based encoder
    Out: intermediate features from 17, 24 and final layer
    '''
    def __init__(self,num_classes = 2):
        super().__init__()
        num_classes = 2
        model1 = models.vgg16(pretrained=True)
        features1, classifier1 = list(model1.features.children()), list(model1.classifier.children())

        self.features1_3 = nn.Sequential(*features1[: 17])
        self.features1_4 = nn.Sequential(*features1[17: 24])
        self.features1_5 = nn.Sequential(*features1[24:])
        
        
    def forward(self,img):
        pool1_3 = self.features1_3(img)
        pool1_4 = self.features1_4(pool1_3)
        pool1_5 = self.features1_5(pool1_4)
        return pool1_3,pool1_4,pool1_5

# Base Encoder 
class encoder_2(nn.Module):
    def __init__(self,num_classes = 2):
        super().__init__()
        num_classes = 2
        model1 = models.vgg16(pretrained=True)
        features1, classifier1 = list(model1.features.children()), list(model1.classifier.children())
        
        self.feature1_2 = nn.Sequential(nn.Conv2d(6,64,kernel_size=3, stride=1, padding=1),nn.ReLU(inplace=True))
        self.features1_3 = nn.Sequential(*features1[2: 17])
        self.features1_4 = nn.Sequential(*features1[17: 24])
        self.features1_5 = nn.Sequential(*features1[24:])
        
        
    def forward(self,img):
        pool1_2 = self.feature1_2(img)
        pool1_3 = self.features1_3(pool1_2)
        pool1_4 = self.features1_4(pool1_3)
        pool1_5 = self.features1_5(pool1_4)
        return pool1_3,pool1_4,pool1_5

# Base Decoder 
class decoder(nn.Module):
    '''
    FCN8 based decoder
    Out: segmentation of ouput with dimension based on the input dimension
    '''
    def __init__(self,n_classes = 2):
        super().__init__()    
        self.score_pool3 = nn.Conv2d(256,n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512,n_classes, kernel_size=1)

        self.upsampling2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4,stride=2, bias=False)
        self.upsampling8 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=16,stride=8, bias=False)

        self.classifier = nn.Sequential(nn.Conv2d(512, n_classes, kernel_size=1), nn.Sigmoid())
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self,pool3,pool4,pool5,x_size):
        o = self.classifier(pool5)
        o = self.upsampling2(o)

        o2 = self.score_pool4(pool4)
        o = o[:, :, 1:1 + o2.size()[2], 1:1 + o2.size()[3]]
        o = o + o2

        o = self.upsampling2(o)

        o2 = self.score_pool3(pool3)
        o = o[:, :, 1:1 + o2.size()[2], 1:1 + o2.size()[3]]
        o = o + o2

        o = self.upsampling8(o)
        cx = int((o.shape[3] - x_size[3]) / 2)
        cy = int((o.shape[2] - x_size[2]) / 2)
        o = o[:, :, cy:cy + x_size[2], cx:cx + x_size[3]]

        return o


# Two stream architecture
class modnet(nn.Module):
    def __init__(self):
        super(modnet,self).__init__()
        num_classes = 2
        self.encoder1 = encoder()
        self.encoder2 = encoder()
        
        self.decoder1 = decoder()
        self.decoder2 = decoder()

    def forward(self, rgb, of):
        x_size = rgb.size()
        # encoder 1
        pool1_3, pool1_4, pool1_5 = self.encoder1(rgb)
        # encoder 2
        pool2_3, pool2_4, pool2_5 = self.encoder2(of)
        # combined features
        pool3 = pool1_3 + pool2_3
        pool4 = pool1_4 + pool2_4
        pool5 = pool1_5 + pool2_5
        # decoder 1
        spatial_out = self.decoder1(pool3,pool4,pool5,x_size)
        #decoder 2
        motion_out = self.decoder2(pool3,pool4,pool5,x_size)
        
        return spatial_out,motion_out

# Two stream architecture
class modnet_2(nn.Module):
    def __init__(self):
        super(modnet_2,self).__init__()
        num_classes = 2
        self.encoder1 = encoder()
        self.encoder2 = encoder_2()
        
        self.decoder1 = decoder()
        self.decoder2 = decoder()

    def forward(self, rgb, of):
        x_size = rgb.size()
        combined_flow = torch.cat((rgb, of),dim=1)
        #print(combined_flow.shape)
        # encoder 1
        pool1_3, pool1_4, pool1_5 = self.encoder1(rgb)
        # encoder 2
        pool2_3, pool2_4, pool2_5 = self.encoder2(combined_flow)
        # combined features
        pool3 = pool1_3 + pool2_3
        pool4 = pool1_4 + pool2_4
        pool5 = pool1_5 + pool2_5
        # decoder 1
        spatial_out = self.decoder1(pool3,pool4,pool5,x_size)
        #decoder 2
        motion_out = self.decoder2(pool3,pool4,pool5,x_size)
        
        return spatial_out,motion_out