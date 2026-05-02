import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from dataloader import *
from loss import *
from utils import *
from network import SHARENetwork_RCA, SHARENetwork_CBAM
import numpy as np
import random
import torch.backends.cudnn as cudnn
import shutil
from loss import loss_all, loss_woARB, loss_woHCM, loss_woContra, loss_woCenter, loss_woKD
import copy
import pdb
import argparse
import warnings
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="code for Muli-Expert")

    parser.add_argument(
        "--local_rank",
        help='local_rank for distributed training',
        type=int,
        default=0,
    )

    parser.add_argument(
        "--lr",
        help='Lr training',
        type=int,
        default=0.001,
    )

    parser.add_argument(
        "--batch_szie",
        help='Barch size training',
        type=int,
        default=16,
    )

    parser.add_argument(
        "--Epochs",
        help='Epochs training',
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        help='Seed training',
        type=int,
        default=666,
    )

    parser.add_argument(
        "--num_classifier",
        help='Num classifier',
        type=int,
        default=2,
    )

    parser.add_argument(
        "--HCM_num",
        help='HCM_num',
        type=int,
        default=9,
    )


    parser.add_argument(
        "--factor",
        help='Factor',
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--factor_hcm",
        help='Factor HCM',
        type=float,
        default=0.6,
    )

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    setup_seed(args.seed)
    local_rank = args.local_rank
    rank = local_rank
    info = {}
    # GPU
    device = torch.device("cuda")
    info['Device'] = device
    devices = torch.cuda.device_count()

    for k, v in args._get_kwargs():
        #print(k, v)
        if k != 'batch_szie':
            info[k] = v

    if devices > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                                init_method='env://')
        info['Batch size'] = args.batch_szie * 2
    else:
        info['Batch size'] = args.batch_szie

    # load whitebait balanced data
    DATASET = '28k'    # dataset = 28k or 34k


    # 4:1:1
    train_data = IMBALANCED_WHITEBAIT(dataset=DATASET)
    valid_data = IMBALANCED_WHITEBAIT_VALID(dataset=DATASET)
    test_data = IMBALANCED_WHITEBAIT_TEST(dataset=DATASET)


    label, num_per_cls, g = train_data.get_label_list(DATASET) # num_per_cls: train
    valid_gt_label, test_gt_label = valid_data.get_gt_label()
    info['Group split'] = g
    # print(majority_minority_split)

    num_classes = len(label)
    in_features = 2048          # resnet101

    net = SHARENetwork_CBAM(in_features, num_classes, args.num_classifier)
    # VERSION = '2expert_CE+HCM+Metric_inter=1_intra=0.0125'
    VERSION = 'Mod_channel_att_Inter=0.05_intra=0.05*0.0125'
    #criterion2 = BGS_V8_RE_L3re()

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4) #optimizer = optim.Adam(net.parameters(),lr=0.0005,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False) #star--, 2023.1.27
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)


    # total_loss = multi_loss(criterion, criterion_ReCE, criterion2, criterion_hcm_upper, device)
    bsce_weight = train_data.get_bsce_weight()
    # total_loss = CE_HCM(bsce_weight, args.factor, args.factor_hcm, args.HCM_num, device)
    total_loss = loss_all(bsce_weight, args.factor, args.factor_hcm, args.HCM_num, device)
    # train num_class_list ,factor, factor_hcm, hcm_num, devic

    # parameters setting
    start_epoch = 1
    epoch = args.Epochs

    info['Epoch'] = epoch

    name = '{}_{}epoch'.format(VERSION, epoch)
    info['Save path'] = name

    net = net.cuda()
    if devices > 1:
        print('Use DDP for training')
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

        train_sampler = torch.utils.data.DistributedSampler(train_data)
        trainloader = DataLoader(train_data, batch_size = args.batch_szie, 
                                shuffle = False, num_workers=4,
                                pin_memory=True, sampler=train_sampler, 
                                drop_last=True
                                )
        valloader = DataLoader(valid_data, batch_size = args.batch_szie, 
                            shuffle = False, num_workers=4, pin_memory=True)
    else:
        trainloader = DataLoader(train_data, batch_size = args.batch_szie, 
                                shuffle = True, num_workers=4,
                                pin_memory=True, drop_last=False
                                )
        valloader = DataLoader(valid_data, batch_size = args.batch_szie, 
                            shuffle = False, num_workers=4, pin_memory=True)

    if not os.path.isdir('./outputs/' + name):
        try:
            os.makedirs('./outputs/' + name)
        except:
            print('create new folder')
    else:
        print('the folder has been exist')
    
    code_folder = './outputs/{}/code'.format(name)
    if not os.path.exists(code_folder):
        os.makedirs(code_folder)
    all_file = os.listdir('./') 
    for i in all_file:
        try:
            file_extension = i.split('.')[1]
            if file_extension == 'py':
                shutil.copy(i, code_folder)
        except:
            continue
    """
    # v1, v2
    train_share(net, trainloader, valloader, epoch, name, start_epoch, optimizer, criterion, criterion2, g)
    """
    logger, log_file = create_logger('./outputs/' + name, info, local_rank)
    warnings.filterwarnings("ignore")

    
    # train_baseline( net, trainloader, valloader, epoch, name, start_epoch, optimizer, criterion, logger)
    # v3
    train_share(net, trainloader, valloader, epoch, name, start_epoch, optimizer, total_loss
                , g, logger, args.num_classifier)
    
    