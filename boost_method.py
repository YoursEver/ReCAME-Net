
from random import gammavariate
import torch
import torch.nn as nn

def edc_fcc(bs, features, labels, num_classes, num_per_cls, gamma=0.1): # feature: backbone後不要view flatten
    tau = [0 for i in range(num_classes)]
    #tau= []
    keys = list(num_per_cls.keys())
    #print(num_per_cls)
    #print(keys)
    for i in range(num_classes): # 計算majority to minority對應的tau,並assign給class
        tau_i = 1 + gamma*(1-(i/num_classes))
        #print(keys[i])
        tau[keys[i]] = round(tau_i, 2)
        #tau.append(round(tau_i, 2))
    #print(tau)
    raw_shappe = features.shape

    tau_batch = []

    for i in range(bs):
        tau_batch.append(tau[labels[i]])

    tau_bath = torch.tensor(tau_batch).cuda()

    tau_bath = tau_bath.view(bs, -1)
    features = features.view(bs, -1)
    new_features = torch.mul(features, tau_bath)
    new_features = new_features.view(raw_shappe)

    return new_features

class FBL_loss(nn.Module):
    def __init__(self, cls_num_list, weight, lambda_=1., classifier = False, gamma=0.):
        super(FBL_loss, self).__init__()
        
        self.num_classes = len(cls_num_list)
        self.weight = weight
        self.classifier = classifier
        self.lambda_ = lambda_
        self.gamma = gamma

        lam_list = torch.cuda.FloatTensor(cls_num_list)
        lam_list = torch.log(lam_list)
        lam_list = lam_list.max() - lam_list
        self.lam_list = lam_list*(1/lam_list.max()) # normalize
    
    #def forward(self, out, labels, cuur):
        

