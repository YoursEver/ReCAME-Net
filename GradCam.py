import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import pdb
class GradCAM:
    def __init__(self, model):
        super(GradCAM, self).__init__()
        self.model = model
        self.feature_maps = []
        self.gradient_maps = []
        self.handlers1 =  []
        self.handlers2 =  []
        #self.hooks = []

        def save_feature_maps(module, input, output):
            #print(output.shape)
            self.feature_maps.append(output)

        def save_gradient_maps(module, grad_in, grad_out):
            #print(grad_out[0].shape)
            self.gradient_maps.append(grad_out[0])

        # for name, module in self.model.named_modules():
        #     if isinstance(module, torch.nn.modules.conv.Conv2d):
        #         self.hooks.append(module.register_forward_hook(save_feature_maps))
        #         self.hooks.append(module.register_backward_hook(save_gradient_maps))
        #print(model.module.backbone[0].layer4)
        module = model.module.backbone[0].layer4 # select layers which want to vis
        
        self.handlers1.append(module.register_forward_hook(save_feature_maps))
        self.handlers1.append(module.register_backward_hook(save_gradient_maps))
        #self.hooks.append(module.register_forward_hook(save_feature_maps))
        #self.hooks.append(module.register_backward_hook(save_gradient_maps))
        ############ module2
        self.feature_maps2 = []
        self.gradient_maps2 = []
        def save_feature_maps2(module, input, output):
            #print(output.shape)
            self.feature_maps2.append(output)

        def save_gradient_maps2(module, grad_in, grad_out):
            #print(grad_out[0].shape)
            self.gradient_maps2.append(grad_out[0])
        
        module2 = model.module.backbone[0].layer4
        self.handlers2.append(module2.register_forward_hook(save_feature_maps2))
        self.handlers2.append(module2.register_backward_hook(save_gradient_maps2))

    def forward(self, x):
        return self.model(x)

    def remove_handlers(self):
        for handle in self.handlers1:
            handle.remove()
        for handle in self.handlers2:
            handle.remove()
       
    
  
    def backward(self, logit, target):
        self.model.zero_grad()
        loss = logit[0, target]
        loss.backward()

        # y = torch.zeros([1, self.model.fc.out_features], dtype=torch.float32).cuda() # (1, class_num)
        # y[0][target] = 1
        # # y.backward()
        # logit.backward(gradient=y)
        #print(self.gradient_maps[0].cpu().numpy().squeeze())
        #grads = self.gradient_maps[0].cpu().detach().numpy() .squeeze()
        #fmap = self.feature_maps[0].cpu().detach().numpy() .squeeze()
        #print(grads.shape) # (2048, 13, 9)
        #print(fmap.shape)

    def get_maps(self):
        feature_maps1 = self.feature_maps[-1].cpu().detach().numpy() .squeeze()
        gradient_maps1 = self.gradient_maps[-1].cpu().detach().numpy() .squeeze()

        feature_maps2 = self.feature_maps2[-1].cpu().detach().numpy() .squeeze()
        gradient_maps2 = self.gradient_maps2[-1].cpu().detach().numpy() .squeeze()
        #print(feature_maps1.shape, feature_maps2.shape, gradient_maps1.shape, gradient_maps2.shape)
        feature_maps = (feature_maps1 + feature_maps2) / 2
        gradient_maps =  (gradient_maps1 + gradient_maps2) / 2
        
        
        tmp = gradient_maps.reshape([gradient_maps.shape[0], -1])
        
        weight = np.mean(tmp, axis=1) # (2048)
        #print((weight.reshape(-1, 1, 1)).shape, feature_maps.shape, weight.shape)
        
        cam = (weight.reshape(-1, 1, 1) * feature_maps).sum(axis=0)
        # cam = cam - np.min(cam)
        # cam = cam / np.max(cam)
        cam = (cam>0)*cam
        cam = (cam/cam.max() ) * 255
        #print(cam.shape)
        return cam

    def display(self, path, cam, correct):
        # 原先path為batch內

        output = path[0].split('/')[-1]
        
        # image =  Image.open(path[0])
        # img_arr = np.array(image)
        image = cv2.imread(path[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cam = cv2.resize(cam, (image.shape[0], image.shape[1]))
        cam = cam.T
        heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
        # print(img_arr.shape)
        # #print(cam.shape)
        #print(heatmap)
        #print(img_arr)
        cam_img = np.uint8(0.7*image + 0.3*heatmap)
        #cam_img = 
        # cv2.imshow('00', cam_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        ele = output.split('.')
       
        output_path = os.path.join('./0905_gradcam', '{}_{}.{}'.format(ele[0], correct, ele[1]))
        #print('******', output_path)
        #breakpoint()
        #cv2.imwrite(output_path, cam_img)