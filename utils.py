import os
import math
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import logging
from evaluate import tsne_function, FusionMatrix

from tqdm import tqdm
import pdb
import csv
from GradCam import GradCAM
from evaluate_CVPR import tsne_function_CVPR
from sklearn.metrics import recall_score
# use GPU

device = torch.device("cuda")

def create_logger(log_path, info, rank=0):
    log_dir = os.path.join(log_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = '{}.log'.format(time_str)
    log_file = os.path.join(log_dir, log_name)
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if rank > 0: 
        return logger, log_file
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info('---------------- Setting as follow ----------------')
    for k, v in info.items():
        logger.info('{}: {}'.format(k, v))
    logger.info('---------------------------------------------------')
    return logger, log_file

def test(net, Loader, criterion, output_path, num_classifier):
    id2cat = {"0": "01_Hm", "1": "01_Se", "2": "01_Su", "3": "01_Sw", "4": "01_Tm", 
              "5": "02_As", "6": "02_De", "7": "02_Em", "8": "02_Gen", "9": "02_Hsp", 
              "10": "02_Sg", "11": "02_Sl", "12": "02_Sz", "13": "03_Im", "14": "04_Ms", 
              "15": "05_Gm", "16": "05_Pa", "17": "05_Pb", "18": "05_Sinsi", "19": "05_Sinter", 
              "20": "05_Sr", "21": "07_Rg", "22": "09_Ar", "23": "11_Eh", "24": "11_Ej", 
              "25": "11_Ep", "26": "11_Si", "27": "11_Tc", "28": "11_Td", "29": "11_Ts", "30": "12_Ema"}

    lt_group = {"Head": ["11_Eh", "11_Ej", "11_Ep", "11_Si"], 'Many': ["01_Tm", "02_De", "02_Sz", "05_Pb", "05_Sr", "11_Tc", "11_Td"],
                "Medium": ["01_Hm", "01_Se", "01_Su", "02_As", "02_Gen", "02_Hsp", "02_Sg", "02_Sl", "05_Gm"], 
                "Few": []}
    #print(g)
    pred_cls_arr = []
    gt_cls_arr = []
    loss_t_arr, loss_CE_arr, loss_hcm_arr = [], [], []
    #label_test_number = []
    tsne, fusion_matrix = tsne_function(31, id2cat), FusionMatrix(31, id2cat)
    tsne_CVPR = tsne_function_CVPR(31, id2cat)
    experts_acc = [[] for i in range(num_classifier)]
    with torch.no_grad():
        #time_test_start = datetime.datetime.now()  
        net.eval()
        fn = nn.Softmax(dim=1)
        start_time = datetime.datetime.now()
        for batch_idx, (inputs, labels, label_group_idx, meta, _) in enumerate(Loader):
            images, labels, label_group_idx = inputs.to(device), labels.to(device), label_group_idx.to(device)

            image_as, label_as, label_group_idxs = [], [], []
            for k in range(num_classifier):
                image_as.append(images)
                label_as.append(labels)
                label_group_idxs.append(label_group_idx)

            outputs_a, feature_a = net(image_as)
            
            w1, w2 = get_delay_weight(24, 0, 1, 90), get_delay_weight(24, 0.01, 0.09, 90)

            loss_t, loss_fc, loss_hcm = criterion(outputs_a, label_as)

           
            loss_t_arr.append(loss_t.item())
            loss_CE_arr.append(loss_fc.item())
            loss_hcm_arr.append(loss_hcm.item())
            current_loss_t = sum(loss_t_arr) / len(loss_t_arr)
            current_loss_ce = sum(loss_CE_arr) / len(loss_CE_arr)
            current_loss_hcm = sum(loss_hcm_arr) / len(loss_hcm_arr)
            # print(len(testLoader))
            avg_logit = torch.mean(torch.stack(outputs_a), dim=0)
            _, predicted = avg_logit.max(1)
            gt_cls_arr.append(label_as[0].detach().cpu().numpy())
            pred_cls_arr.append(predicted.detach().cpu().numpy())

            for k in range(num_classifier):
                _, expert_pred = outputs_a[k].max(1)
                experts_acc[k].append(expert_pred.detach().cpu().numpy())
             
            current_acc_expert = [0] * num_classifier
            for k in range(num_classifier):
                current_acc_expert[k] = np.round(accuracy(experts_acc[k], gt_cls_arr), 4)
                
            current_acc = np.round(accuracy(pred_cls_arr, gt_cls_arr), 4)


            avg_feat = torch.mean(torch.stack(feature_a), dim=0)
            # tsne.store_data(avg_feat, labels)
            tsne_CVPR.store_data(avg_feat, labels)
            fusion_matrix.update(predicted.cpu().numpy(), labels.detach().cpu().numpy())

            ###### 輸出 feature csv
            # x_feature = torch.cat((x_feature, labels.unsqueeze(1), labels.unsqueeze(1)), 1)
            # feat_arr.append(x_feature.cpu().numpy())


            #break 
            if (batch_idx) % 20 == 0:
                print('Batch: {}/{} Current_Acc: {} Loss_t: {} Loss_ce: {} Loss_hcm: {}'.format(batch_idx, len(Loader), current_acc, 
                                                                            current_loss_t, current_loss_ce, current_loss_hcm))
        # feat_arr = np.concatenate(feat_arr, axis=0)
        # print(feat_arr.shape)

        # with open('./Yume_feature_STAR.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(feat_arr)


        end_time = datetime.datetime.now()
        print('--------------- Total time: {} ---------------'.format(end_time-start_time))

        print(current_acc_expert)
        print('Final Current_Acc: {} Loss_t: {} Loss_ce: {} Loss_hcm: {}'.format(current_acc, 
                                                                            current_loss_t, current_loss_ce, current_loss_hcm))
        
        #output_path = os.path.join(output_path, 'val_result')
        #fusion_matrix.plot_bar_rec(output_path=output_path, LT_group=lt_group)
        
        #g = [[23, 24, 25, 26], [4, 6, 12, 17, 20, 27, 28], [0, 1, 2, 5, 8, 9, 10, 11, 15], [3, 7, 13, 14, 16, 18, 19, 21, 22, 29, 30]]
        # tsne.visualize(output_path)
        #tsne_CVPR.tsne_pca()
        #tsne_CVPR.visualize_4head(output_path, g)
        #g = [[23], [24, 25, 26], 
             #[0, 4, 6, 8, 9, 11, 12, 15, 17, 20, 27, 28], [1, 2, 3, 5, 7, 10, 13, 14, 16, 18, 19, 21, 22, 29, 30]]
        #tsne_CVPR.visualize_4shot(output_path, g)
        
        #fusion_matrix.plot_confusion_matrix(acc=current_acc, output_path=output_path, normalize=False)
    
def accuracy(pred_cls, gt_cls):
    pred_cls = np.concatenate(pred_cls, axis=0)
    gt_cls = np.concatenate(gt_cls, axis=0)
    total_len = 0
    correct = 0
    for i in range(len(pred_cls)):
        total_len += 1
        if pred_cls[i] == gt_cls[i]:
            correct += 1
        else:
            continue
    return (correct/total_len) * 100

    # pred_cls = np.concatenate(pred_cls, axis=0)
    # gt_cls = np.concatenate(gt_cls, axis=0)

    # return recall_score(gt_cls, pred_cls, average='micro')
        

def get_delay_weight(t, w0, wt, T):
    if t < T/3:
        return w0
    elif t >= T/3 and t < (2*T)/3:
        return w0 + (t-(T/3))*(wt*3/T)
    else:
        return wt
# v3
from torch.distributions import Categorical
def train_share(net, trainloader, validloader, epochs, weight_name, start_epoch, optimizer, criterion
                , g, logger, num_classifier):
    time_train_start = datetime.datetime.now()

    best_result, best_epoch = 0, 0
    
    #===below===star++, 2023.1.16
    m_delayed_rebalacing_factor=0
    #===above===star++, 2023.1.16
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200, eta_min=1e-4
            )
    for epoch in range(start_epoch, start_epoch+epochs):
        # print('\nEpoch: %d' % (epochs))
        net.train()

        train_loss_t, train_loss_CE, train_loss_hcm, train_loss_metric = [], [], [], []
        train_loss_rwCE = []
        train_pred_arr, train_gt_arr = [], []
        
       
        #===above===star++, 2023.1.16
        scheduler.step()
        logger.info('Epoch: {} Lr: {}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        experts_acc = [[] for i in range(num_classifier)]
        for batch_idx, (inputs, labels, label_group_idx, meta) in enumerate(trainloader):
            image_a, label_a, label_group_idx = inputs.to(device), labels.to(device), label_group_idx.to(device)
            image_b, label_b = meta['sample_image'].to(device), meta['sample_label'].to(device)
            #print(label_a, label_b)
            #print(k)
            image_as, label_as, label_group_idxs = [], [], []
           
            for k in range(num_classifier):
                image_as.append(image_a)
                label_as.append(label_a)
                label_group_idxs.append(label_group_idx)

            
            outputs_a, feature_a = net(image_as)
 
            
            # w1, w2 = get_delay_weight(epochs, 0, 1, epoch), get_delay_weight(epochs, 0.01, 0.09, epoch)
            #print(len(label_group_idxs), )
            loss, loss_fc, loss_rwce, loss_hcm_t, loss_metric = criterion(outputs_a, label_as, feature_a)

            ####### store train loss ########
            train_loss_t.append(loss.item())
            train_loss_CE.append(loss_fc.item())
            train_loss_hcm.append(loss_hcm_t.item())
            train_loss_metric.append(loss_metric.item())
            train_loss_rwCE.append(loss_rwce.item())
            
            ##################################

            optimizer.zero_grad() # 清空過往梯度
            loss.backward()      # 計算當前梯度
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)     # 避免 loss inf / nan 做梯度裁減 #star-- 2023.01.28
            optimizer.step()     # 根據梯度更新網路參數

#            train_loss += loss.item() #star--,2023.1.30 把這行註解掉看看會不會快一點
            
            # upper fc branch acc
            avg_logit = torch.mean(torch.stack(outputs_a), dim=0)
            _, predicted = avg_logit.max(1)
            train_gt_arr.append(label_as[0].detach().cpu().numpy())
            train_pred_arr.append(predicted.detach().cpu().numpy())
            

            for k in range(num_classifier):
                _, expert_pred = outputs_a[k].max(1)
                experts_acc[k].append(expert_pred.detach().cpu().numpy())
           
            current_acc_expert = [0] * num_classifier
            for k in range(num_classifier):
                current_acc_expert[k] = np.round(accuracy(experts_acc[k], train_gt_arr), 4)

            current_acc = np.round(accuracy(train_pred_arr, train_gt_arr), 4)
            current_loss_t = np.round(sum(train_loss_t) / len(train_loss_t), 4)
            current_loss_CE = np.round(sum(train_loss_CE) / len(train_loss_CE), 4)
            current_loss_hcm = np.round(sum(train_loss_hcm)/len(train_loss_hcm), 4)
            current_loss_metric = np.round(sum(train_loss_metric) / len(train_loss_metric), 4)
            current_loss_rwCE = np.round(sum(train_loss_rwCE) / len(train_loss_rwCE), 4)
                
            if batch_idx % 100 == 0:
                #print('*******', batch_idx)
                logger.info('Each experts: {}'.format(current_acc_expert))
                logger.info('Epoch: {} Batch {}/{} Loss_t: {} Loss_CE: {} Loss_rwCE: {} loss_hcm:{} loss_metric: {} Acc: {}'.format(epoch, 
                                                                 batch_idx, len(trainloader), current_loss_t, current_loss_CE, current_loss_rwCE, 
                                                                current_loss_hcm, current_loss_metric, current_acc ))
                #break 
#        print('training label mean accuracy of %d epoch: FC %.3f%% | BGS %.3f%%' % ( epochs+1, sum(acc_list0)/len(acc_list0), sum(acc_list)/len(acc_list)) )
#        print('training min accuracy of %d epoch: FC %.3f%%, FC max accuracy: %.3f%% | BGS %.3f%%, BGS max accuracy: %.3f%%' % (epochs+1, min(acc_list0), max(acc_list0), min(acc_list), max(acc_list)) )
        
        logger.info('-------------------------- End of {} epoch Training --------------------------'.format(epoch))
        logger.info('--------------- Each experts: {} ---------------'.format(current_acc_expert))
        logger.info('------- Epoch: {} Loss_t: {} Loss_CE: {} Loss_rwCE: {} loss_hcm:{} loss_metric: {} Acc: {} -------'.format(epoch,current_loss_t, 
                                                    current_loss_CE, current_loss_rwCE, current_loss_hcm, current_loss_metric, current_acc))
        if (epoch) % 2 == 0:
            torch.save(net.state_dict(), './outputs/{}/Epoch={}.pth'.format(weight_name, epoch))

        with torch.no_grad():
            net.eval()        

            val_loss_t, val_loss_CE, val_loss_hcm, val_loss_metric, val_loss_rwCE = [], [], [], [], []
            val_pred_arr, val_gt_arr = [], []
            experts_acc_v = [[] for i in range(num_classifier)]
            for batch_idx_, (inputs_valid, labels_valid, label_group_idx_val, meta_valid) in enumerate(validloader):
                image_a_v, label_a_v, label_group_idx_v = inputs_valid.to(device), labels_valid.to(device), label_group_idx_val.to(device)
                image_b_v, label_b_v = meta_valid['sample_image'].to(device), meta_valid['sample_label'].to(device)
                #print(label_a, label_b)
                #print(k)
                image_as_v, label_as_v, label_group_idxs_v = [], [], []
                
                for k in range(num_classifier):
                    image_as_v.append(image_a_v)
                    label_as_v.append(label_a_v)
                    label_group_idxs_v.append(label_group_idx_v)
                   
                outputs_a_v, feature_a_v = net(image_as_v)
               
                # w1, w2 = get_delay_weight(epochs, 0, 1, epoch), get_delay_weight(epochs, 0.01, 0.09, epoch)
                #print(len(label_group_idxs), )
                loss_v, loss_fc_v, loss_rwCE_v, loss_hcm_t_v, loss_metric_v = criterion(outputs_a_v, label_as_v, feature_a_v,
                                                                           )

                ####### store train loss ########
                val_loss_t.append(loss_v.item())
                val_loss_CE.append(loss_fc_v.item())
                val_loss_hcm.append(loss_hcm_t_v.item())
                val_loss_metric.append(loss_metric_v.item())
                val_loss_rwCE.append(loss_rwCE_v.item())
                ##################################

                # upper fc branch acc
                avg_logit_v = torch.mean(torch.stack(outputs_a_v), dim=0)
                _, predicted_v = avg_logit_v.max(1)
                val_gt_arr.append(label_as_v[0].detach().cpu().numpy())
                val_pred_arr.append(predicted_v.detach().cpu().numpy())

                for k in range(num_classifier):
                    _, expert_pred = outputs_a_v[k].max(1)
                    experts_acc_v[k].append(expert_pred.detach().cpu().numpy())
            
                current_acc_expert = [0] * num_classifier
                for k in range(num_classifier):
                    current_acc_expert[k] = np.round(accuracy(experts_acc_v[k], val_gt_arr), 4)
                
                current_acc_v = np.round(accuracy(val_pred_arr, val_gt_arr), 4)
                current_loss_t_v = np.round(sum(val_loss_t) / len(val_loss_t), 4)
                current_loss_CE_v = np.round(sum(val_loss_CE) / len(val_loss_CE), 4)
                current_loss_hcm_v = np.round(sum(val_loss_hcm)/len(val_loss_hcm), 4)
                current_loss_metric_v = np.round(sum(val_loss_metric)/len(val_loss_metric), 4)
                current_loss_rwCE_v = np.round(sum(val_loss_rwCE)/len(val_loss_rwCE), 4)


                if batch_idx % 100 == 0:
                    logger.info('Each experts: {}'.format(current_acc_expert))
                    logger.info('Epoch: {} Batch {}/{} Loss_t: {} Loss_CE: {} Loss_rwCE: {} loss_hcm:{} loss_metric: {} Acc: {}'.format(epoch, 
                                                                    batch_idx_, len(validloader), current_loss_t_v, current_loss_CE_v, current_loss_rwCE_v,
                                                                    current_loss_hcm_v, current_loss_metric_v, current_acc_v ))
                    #break 
    #        print('training label mean accuracy of %d epoch: FC %.3f%% | BGS %.3f%%' % ( epochs+1, sum(acc_list0)/len(acc_list0), sum(acc_list)/len(acc_list)) )
    #        print('training min accuracy of %d epoch: FC %.3f%%, FC max accuracy: %.3f%% | BGS %.3f%%, BGS max accuracy: %.3f%%' % (epochs+1, min(acc_list0), max(acc_list0), min(acc_list), max(acc_list)) )
            
                    #break 
            
        
            if current_acc_v > best_result:
                best_result, best_epoch = current_acc_v, epoch
                best_branch = 'fc'
                torch.save({ 
                        'state_dict': net.state_dict(),
                        'epoch': epoch,
                        'best_result': best_result,
                        'best_epoch': best_epoch,
                    }, './outputs/{}/best_model.pth'.format(weight_name)  
                        #os.path.join(model_dir, "best_model.pth")
                )
         

            logger.info('-------------------------------- Val Epoch: {} --------------------------------'.format(epoch))
            logger.info('--------------- Each experts: {} ---------------'.format(current_acc_expert))
            logger.info('------------ Epoch: {} Loss_t: {} Loss_CE: {} Loss_rwCE: {} loss_hcm:{} loss_metric: {} Acc: {} ------------'.format(epoch, 
                                                                    current_loss_t_v, current_loss_CE_v, current_loss_rwCE_v, 
                                                                    current_loss_hcm_v, current_loss_metric_v, current_acc_v ))
            
            logger.info('-------------------- Best Epoch: {} Best_Branch: {} Best_Acc: {} --------------------'.format(best_epoch, best_branch, 
                                                                                                                       best_result))
                
    time_train_end = datetime.datetime.now()
    print("Training finished...")
    print("Training cost ", time_train_end - time_train_start)
    print("BEST MODEL IS %s BRANCH AND THE VALID ACCURACY IS %.3f%% AT %d epoch" % (best_branch, best_result, best_epoch))
    


id2cat = {"0": "01_Hm", "1": "01_Se", "2": "01_Su", "3": "01_Sw", "4": "01_Tm", 
              "5": "02_As", "6": "02_De", "7": "02_Em", "8": "02_Gen", "9": "02_Hsp", 
              "10": "02_Sg", "11": "02_Sl", "12": "02_Sz", "13": "03_Im", "14": "04_Ms", 
              "15": "05_Gm", "16": "05_Pa", "17": "05_Pb", "18": "05_Sinsi", "19": "05_Sinter", 
              "20": "05_Sr", "21": "07_Rg", "22": "09_Ar", "23": "11_Eh", "24": "11_Ej", 
              "25": "11_Ep", "26": "11_Si", "27": "11_Tc", "28": "11_Td", "29": "11_Ts", "30": "12_Ema"}

lt_group = {"Head": ["11_Eh", "11_Ej", "11_Ep", "11_Si"], 'Many': ["01_Tm", "02_De", "02_Sz", "05_Pb", "05_Sr", "11_Tc", "11_Td"],
                "Medium": ["01_Hm", "01_Se", "01_Su", "02_As", "02_Gen", "02_Hsp", "02_Sg", "02_Sl", "05_Gm"], 
                "Few": []}

def gradcam(net, loader, g):
  
    grad_cam = GradCAM(net)
    count = 0
    net.eval()

    for batch_idx, (inputs, labels, label_group_idx, meta, path) in enumerate(tqdm(loader)):
        # if labels not in g[2]:
        #     continue;

    
        inputs, labels = inputs.to(device), labels.to(device=device, dtype=torch.int64)
        label_group_idx = label_group_idx.to(device)

        image_as = []
                
        for k in range(2):
            image_as.append(inputs)

        #breakpoint()
  
        x_fc, feature_a_v = grad_cam.forward(image_as) 
        
        #x_fc, feature_a_v
        #breakpoint()
      
        avg_logit_v = torch.mean(torch.stack(x_fc), dim=0)

        pred_cls = torch.argmax(avg_logit_v, dim=1)
        #print(x_fc[0].shape)
        #pred_cls = torch.argmax(x_fc[0], dim=1)
        #print(pred_cls, labels, avg_pred_cls)
        #pred_cls = x_fc[0]
        #print(pred_cls2, labels)
        #if pred_cls[0].item() == labels[0].item():
            #loss = x_fc[labels]
        correct = pred_cls.item() == labels.item()
        #print(correct)
        grad_cam.backward(avg_logit_v, labels)
        
        cam = grad_cam.get_maps()
        
        grad_cam.display(path, cam, correct)
        grad_cam.remove_handlers()

        count += 1
 

    print('Count: {} '.format(count))



