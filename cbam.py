import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM_module(nn.Module):
    def __init__(self, channels, reduction, pool_types=['avg', 'max'], channel=False, spatial=False):
        super(CBAM_module, self).__init__()
        self.spatial = spatial
        self.channel = channel
        if channel:
            self.channel_att = channel_att_module(channels, reduction, pool_types)
        if spatial:
            self.spatial_att = spatial_att_module(7)

        #print(self.spatial, self.channel)
    def forward(self, x):
        if self.channel:
            #print('Use Channel')
            x = self.channel_att(x)

        if self.spatial:
            #print('Use Spatial')
            x = self.spatial_att(x)
        #print(x.shape)
        return x

class spatial_att_module(nn.Module):
    def __init__(self, kernel_size):
        super(spatial_att_module, self).__init__()
        #  2, 1, 7, stride=1, padding=(kernel_size-1) // 2, dilation=1, groups=1, relu=False, bn=True, bias=False
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2, 
                              dilation=1, groups=1, bias=False)
        
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        cat_feats = self.compress(x) # (bs, 2, H, W)
        x_out = self.conv(cat_feats)
        x_out = self.bn(x_out)
        weight = self.activation(x_out)
        return x * weight

    def compress(self, x):
        max_feat = torch.max(x, dim=1)[0].unsqueeze(1) # (bs, 1, H, W)
        avg_feat = torch.mean(x, dim=1).unsqueeze(1) # (bs, 1, H, W)
        cat_feats = torch.cat((max_feat, avg_feat), dim=1) # (bs, 2, H, W)
        return cat_feats

class channel_att_module(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(channel_att_module, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels//reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels//reduction_ratio, gate_channels),
        )

        self.pool_types = pool_types
        self.activation = nn.Sigmoid()
    
    def forward(self, x):

        for pool_type in self.pool_types:
            w, h = x.size(2), x.size(3)
            if pool_type == 'avg':
                avg_maps = F.avg_pool2d(x, kernel_size=(w, h), stride=(w, h))
                avg_maps = self.mlp(avg_maps)

            elif pool_type == 'max':
                max_maps = F.max_pool2d(x, kernel_size=(w, h), stride=(w, h))
                max_maps = self.mlp(max_maps)
            

        mix_feats = avg_maps + max_maps
        weight = self.activation(mix_feats).unsqueeze(2).unsqueeze(3).expand_as(x)
        #print('x: {} Ch weight: {}'.format(x.shape, weight.shape))

        return x * weight