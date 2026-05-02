import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
import copy
from resnet_CBAM import res101_CBAM
from resnet_RCA import res101_RCA
class Cos_Classifier(nn.Module):
    """ plain cosine classifier """
    # 31, 2048, 16, True
    def __init__(self, num_classes=10, in_dim=640, scale=16, bias=False): # scale=16, bias=True
        super(Cos_Classifier, self).__init__()
        self.scale = scale
        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        self.init_weights() # init bias=0, weight: uniform distribution

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True) # torch.norm (x, p, dim) along dim=1 do 2 norm
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        #print(ex.shape)
        #print(self.weight.shape)
        out = torch.mm(ex, self.scale * ew.t()) + self.bias # do y=wx+b, torch.mm: matrix multiple
        return out
# v2 
class SHARENetwork_RCA(nn.Module):
    def __init__(self, in_features, num_classes, num_classifier):
        super(SHARENetwork_RCA, self).__init__()
 
        self.in_features = in_features
       
        self.backbone = nn.ModuleList(
                res101_RCA(
                    last_layer_stride=2,
                    pretrained_model='/home/nvlab110/SDB1/nvlab110hao/whitebait/resnet101_v1.pth',
                    mod_channel = True,
                ) for i in range(num_classifier))

      
        self.pool = nn.ModuleList(nn.AdaptiveAvgPool2d((1, 1)) for i in range(num_classifier))
        #self.backbone = nn.Sequential(*list(original_model.children())[:])    # resnet32
        #self.conv2 = nn.Conv2d(2048, 64, kernel_size=(7, 7), bias=False)
        
        self.fc = nn.ModuleList(Cos_Classifier(num_classes, in_features, scale=16, bias=True) 
                                for i in range(num_classifier))
        
        
    def forward(self, inputs):
       
        xs, feats = [], []
        for i in range(len(inputs)):
            #print(i)
            x = (self.backbone[i])(inputs[i])
            x_feature = (self.pool[i])(x)
            x_feature = x_feature.view(x_feature.size(0), -1)
 
            feats.append(x_feature)
            x_fc = (self.fc[i])(x_feature)

            xs.append(x_fc)

        return xs, feats

class SHARENetwork_CBAM(nn.Module):
    def __init__(self, in_features, num_classes, num_classifier):
        super(SHARENetwork_CBAM, self).__init__()
 
        self.in_features = in_features
       
        self.backbone = nn.ModuleList(
                res101_CBAM(
                    last_layer_stride=2,
                    pretrained_model='/home/nvlab110/SDB1/nvlab110hao/whitebait/resnet101_v1.pth',
                    spatial=True,
                    channel=True,
                ) for i in range(num_classifier))

      
        self.pool = nn.ModuleList(nn.AdaptiveAvgPool2d((1, 1)) for i in range(num_classifier))
        #self.backbone = nn.Sequential(*list(original_model.children())[:])    # resnet32
        #self.conv2 = nn.Conv2d(2048, 64, kernel_size=(7, 7), bias=False)
        
        self.fc = nn.ModuleList(Cos_Classifier(num_classes, in_features, scale=16, bias=True) 
                                for i in range(num_classifier))
        
        
    def forward(self, inputs):
       
        xs, feats = [], []
        for i in range(len(inputs)):
            #print(i)
            x = (self.backbone[i])(inputs[i])
            x_feature = (self.pool[i])(x)
            x_feature = x_feature.view(x_feature.size(0), -1)
 
            feats.append(x_feature)
            x_fc = (self.fc[i])(x_feature)

            xs.append(x_fc)

        return xs, feats