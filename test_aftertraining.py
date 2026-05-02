import os
import torch
import torch.nn as nn
import torchvision.models as models
from dataloader import *
from torch.utils.data import DataLoader
from utils import *
from network import *
from loss_0725 import *
import copy
from inference_csv import inference_wo_label, inference_multi_expert
# whitebait

BATCH_SIZE = 64
num_classifier = 2
DATASET = '28k'    # dataset = 28k or 34k

device = torch.device("cuda")
# 4:1:1
train_data = IMBALANCED_WHITEBAIT(dataset=DATASET)
trainloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers=4)
valid_data = IMBALANCED_WHITEBAIT_VALID(dataset=DATASET)
valloader = DataLoader(valid_data, batch_size = BATCH_SIZE, shuffle = False, num_workers=4)
test_data = IMBALANCED_WHITEBAIT_TEST(dataset=DATASET)
testloader  = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False, num_workers=4)

label, num_per_cls, g = train_data.get_label_list(DATASET) # num_per_cls: train
valid_gt_label, test_gt_label = valid_data.get_gt_label()



num_classes = len(label)
in_features = 2048          # resnet101
#info['Pre-train_weight_path'] = pretrain_weight_path

net = SHARENetwork(in_features, num_classes, num_classifier)
net.cuda()
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net).cuda()


checkpoints_path = './best_model.pth'
checkpoints1 = torch.load(checkpoints_path)
net.load_state_dict(checkpoints1['state_dict'])

# checkpoints_path = './Epoch=50.pth'
# checkpoints1 = torch.load(checkpoints_path)
# net.load_state_dict(checkpoints1)

tmp_output_path = checkpoints_path.split('/')[0:-1]
# print(tmp_output_path)
output_path = ''
for i in tmp_output_path:
    output_path = os.path.join(output_path, i)
# print(output_path)

bsce_weight = train_data.get_bsce_weight()
total_loss = CE_HCM(bsce_weight, 0.6, 0.6, 9, device)
#print('Grouping: ', g)
test(net, testloader, total_loss, output_path, num_classifier)
# gradcam(net, testloader, g)
# test_gen_h5file(net, testLoader)
# test(net, validLoader, criterion, criterion2, output_path)

#inference_multi_expert(net, testloader, './')
