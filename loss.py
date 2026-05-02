from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import datetime
class Contrastive_Loss_for_Upper_Branch(nn.Module):
    def __init__(self, device):
        super(Contrastive_Loss_for_Upper_Branch, self).__init__()
        self.device = device
        return
    
    def forward(self, label_a, feature_a):
    #def forward(self, label_a, label_b, pred_a, pred_b):
        #label_a, label_b, pred_a, pred_b = label_a.cuda(), label_b.cuda(), pred_a.cuda(), pred_b.cuda()
        # v1
        contr, mycounter = 0, 0
        for i in range(len(label_a)-1):
            for j in range(i+1, len(label_a)):
                if label_a[i] == label_a[j]:  # same class
                    contr += torch.linalg.norm( feature_a[i] - feature_a[j], dim=0, ord=2)                
                    mycounter +=1
        # contr = 2
        loss = (contr/len(label_a))/len(label_a)
        if torch.is_tensor(loss):
            return loss
        else:
            return torch.tensor(loss, dtype=torch.float32, requires_grad=True).to(self.device)



class ContrLoss(nn.Module): # inter
    def __init__(self):
        super(ContrLoss, self).__init__()
        return
    
    def forward(self, label_a, feature_a):
    #def forward(self, label_a, label_b, pred_a, pred_b):
        #label_a, label_b, pred_a, pred_b = label_a.cuda(), label_b.cuda(), pred_a.cuda(), pred_b.cuda()
        contr = 0
        # v1
        for i in range(len(label_a)-1):
            for j in range(i+1, len(label_a)):
                feat_dis = torch.linalg.norm( feature_a[i] - feature_a[j], dim=0, ord=2)         
                if label_a[i] == label_a[j]:
                    contr += feat_dis
                else:
                    contr += torch.max(torch.tensor(0).cuda(), 20-feat_dis)
                
        
        # for i in range(len(label_a)):
        #     for j in range(len(label_b)):
        #         if label_a[i] - label_b[j] == 0:  # same class
        #             Y = torch.tensor(1)
        #         else:
        #             Y = torch.tensor(0)
        #         contr += Y * torch.linalg.norm( pred_a[i] - pred_b[j], dim=0, ord=2) + (1-Y) * torch.max(torch.tensor(0).cuda(), 0.5-torch.linalg.norm( pred_a[i] - pred_b[j], dim=0, ord=2))
        
        # v2
        # matrix_distance = torch.linalg.norm(feature_a - feature_b, dim=1, ord=2)
        # #same_class = (label_a == label_b)*matrix_distance
        # #diff_class = (label_a != label_b)*(torch.max(torch.tensor(0).cuda(), 20-matrix_distance))
        # contr = sum( (label_a == label_b)*matrix_distance + (label_a != label_b)*(torch.max(torch.tensor(0).cuda(), 20-matrix_distance)) ) 
        #print(contr)
        return (contr/len(label_a))/len(label_a)


def NBOD(inputs, factor):

    classifier_num = len(inputs)
    if classifier_num == 1:
        return 0
    
    logits_softmax = []
    logits_logsoftmax = []
    for i in range(classifier_num):
        logits_softmax.append(F.softmax(inputs[i], dim=1))
        logits_logsoftmax.append(torch.log(logits_softmax[i] + 1e-9))

    loss_mutual = 0
    for i in range(classifier_num):
        for j in range(classifier_num):
            if i == j:
                continue
            # F.kl_div(被指導的log(p), 指導者的p)
            loss_mutual += factor * F.kl_div(logits_logsoftmax[i], logits_softmax[j],reduction='batchmean')
    loss_mutual /= (classifier_num - 1)
    #print(loss_mutual)
    return  loss_mutual

class loss_all(nn.Module):
    def __init__(self, num_class_list, factor, factor_hcm, hcm_num, device):
        super(loss_all, self).__init__()
        print('************ All loss ****************')
        self.bsce_weight = torch.FloatTensor(num_class_list).to(device)
        self.factor = factor
        self.factor_hcm = factor_hcm
        self.hcm_n = hcm_num
        self.intra_loss = Contrastive_Loss_for_Upper_Branch(device)
        self.inter_loss = ContrLoss()

    def get_delay_weight(self, t, w0, wt, T):
        if t < T/3:
            return w0
        elif t >= T/3 and t < (2*T)/3:
            return w0 + (t-(T/3))*(wt*3/T)
        else:
            return wt

    def forward(self, inputs, targets_a, feats_a):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        #w2 = self.get_delay_weight(epoch, 0.01, 0.09, epochs)

        classifier_num = len(inputs)
        loss_HCM = 0
        loss = 0
        los_ce = 0
        loss_inter = 0
        loss_intra = 0

        inputs_HCM_balance = []
        inputs_balance = []
        class_select = inputs[0].scatter(1, targets_a[0].unsqueeze(1), 999999)
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_n]
        mask = torch.zeros_like(inputs[0]).scatter(1, class_select_include_target, 1)
        for i in range(classifier_num):

            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)
            inputs_HCM_balance.append(logits * mask)

            los_ce += F.cross_entropy(logits, targets_a[0])
            loss_HCM += F.cross_entropy(inputs_HCM_balance[i], targets_a[0])
            loss_inter += self.inter_loss(targets_a[0], feats_a[i])
            loss_intra += self.intra_loss(targets_a[0], feats_a[i])

        loss += NBOD(inputs_balance, factor=self.factor)
        loss += NBOD(inputs_HCM_balance, factor=self.factor_hcm)
        # loss += los_ce + loss_HCM + 0.0125*loss_intra
        #loss += los_ce + loss_HCM + 0.05*loss_inter + 0.0125*loss_intra
        loss += los_ce + loss_HCM + 0.05*loss_inter + 0.05*0.0125*loss_intra
        # loss += los_ce + loss_HCM

        return loss, los_ce, los_ce, loss_HCM, 0.05*loss_inter + 0.05*0.0125*loss_intra
 
class loss_woARB(nn.Module):

    def __init__(self, num_class_list, factor, factor_hcm, hcm_num, device):
        super(loss_woARB, self).__init__()
        print('************ All loss ****************')
        self.bsce_weight = torch.FloatTensor(num_class_list).to(device)
        self.factor = factor
        self.factor_hcm = factor_hcm
        self.hcm_n = hcm_num
        self.intra_loss = Contrastive_Loss_for_Upper_Branch(device)
        self.inter_loss = ContrLoss()

    def get_delay_weight(self, t, w0, wt, T):
        if t < T/3:
            return w0
        elif t >= T/3 and t < (2*T)/3:
            return w0 + (t-(T/3))*(wt*3/T)
        else:
            return wt

    def forward(self, inputs, targets_a, feats_a):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        #w2 = self.get_delay_weight(epoch, 0.01, 0.09, epochs)

        classifier_num = len(inputs)
        loss_HCM = 0
        loss = 0
        los_ce = 0
        loss_inter = 0
        loss_intra = 0

        inputs_HCM_balance = []
        inputs_balance = []
        class_select = inputs[0].scatter(1, targets_a[0].unsqueeze(1), 999999)
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_n]
        mask = torch.zeros_like(inputs[0]).scatter(1, class_select_include_target, 1)
        for i in range(classifier_num):

            # logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            logits = inputs[i]
            inputs_balance.append(logits)
            inputs_HCM_balance.append(logits * mask)

            los_ce += F.cross_entropy(logits, targets_a[0])
            loss_HCM += F.cross_entropy(inputs_HCM_balance[i], targets_a[0])
            loss_inter += self.inter_loss(targets_a[0], feats_a[i])
            loss_intra += self.intra_loss(targets_a[0], feats_a[i])

        loss += NBOD(inputs_balance, factor=self.factor)
        loss += NBOD(inputs_HCM_balance, factor=self.factor_hcm)
        # loss += los_ce + loss_HCM + 0.0125*loss_intra
        #loss += los_ce + loss_HCM + 0.05*loss_inter + 0.0125*loss_intra
        loss += los_ce + loss_HCM + 0.05*loss_inter + 0.05*0.0125*loss_intra
        # loss += los_ce + loss_HCM

        return loss, los_ce, los_ce, loss_HCM, 0.05*loss_inter + 0.05*0.0125*loss_intra
    
class loss_woHCM(nn.Module):
    def __init__(self, num_class_list, factor, factor_hcm, hcm_num, device):
        super(loss_woHCM, self).__init__()
        print('************ Loss woHCM ****************')
        self.bsce_weight = torch.FloatTensor(num_class_list).to(device)
        self.factor = factor
        self.factor_hcm = factor_hcm
        self.hcm_n = hcm_num
        self.intra_loss = Contrastive_Loss_for_Upper_Branch(device)
        self.inter_loss = ContrLoss()

    def get_delay_weight(self, t, w0, wt, T):
        if t < T/3:
            return w0
        elif t >= T/3 and t < (2*T)/3:
            return w0 + (t-(T/3))*(wt*3/T)
        else:
            return wt

    def forward(self, inputs, targets_a, feats_a):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        #w2 = self.get_delay_weight(epoch, 0.01, 0.09, epochs)

        classifier_num = len(inputs)
        loss_HCM = 0
        loss = 0
        los_ce = 0
        loss_inter = 0
        loss_intra = 0

        inputs_HCM_balance = []
        inputs_balance = []
        class_select = inputs[0].scatter(1, targets_a[0].unsqueeze(1), 999999)
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_n]
        mask = torch.zeros_like(inputs[0]).scatter(1, class_select_include_target, 1)
        for i in range(classifier_num):

            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)
            inputs_HCM_balance.append(logits * mask)

            los_ce += F.cross_entropy(logits, targets_a[0])
            loss_HCM += F.cross_entropy(inputs_HCM_balance[i], targets_a[0])
            loss_inter += self.inter_loss(targets_a[0], feats_a[i])
            loss_intra += self.intra_loss(targets_a[0], feats_a[i])

        loss += NBOD(inputs_balance, factor=self.factor)
        #loss += NBOD(inputs_HCM_balance, factor=self.factor_hcm)

        loss += los_ce + 0.05*loss_inter + 0.05*0.0125*loss_intra
  

        return loss, los_ce, los_ce, loss_HCM, 0.05*loss_inter + 0.05*0.0125*loss_intra

class loss_woContra(nn.Module):
    def __init__(self, num_class_list, factor, factor_hcm, hcm_num, device):
        super(loss_woContra, self).__init__()
        print('************ All loss ****************')
        self.bsce_weight = torch.FloatTensor(num_class_list).to(device)
        self.factor = factor
        self.factor_hcm = factor_hcm
        self.hcm_n = hcm_num
        self.intra_loss = Contrastive_Loss_for_Upper_Branch(device)
        self.inter_loss = ContrLoss()

    def get_delay_weight(self, t, w0, wt, T):
        if t < T/3:
            return w0
        elif t >= T/3 and t < (2*T)/3:
            return w0 + (t-(T/3))*(wt*3/T)
        else:
            return wt

    def forward(self, inputs, targets_a, feats_a):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        #w2 = self.get_delay_weight(epoch, 0.01, 0.09, epochs)

        classifier_num = len(inputs)
        loss_HCM = 0
        loss = 0
        los_ce = 0
        loss_inter = 0
        loss_intra = 0

        inputs_HCM_balance = []
        inputs_balance = []
        class_select = inputs[0].scatter(1, targets_a[0].unsqueeze(1), 999999)
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_n]
        mask = torch.zeros_like(inputs[0]).scatter(1, class_select_include_target, 1)
        for i in range(classifier_num):

            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)
            inputs_HCM_balance.append(logits * mask)

            los_ce += F.cross_entropy(logits, targets_a[0])
            loss_HCM += F.cross_entropy(inputs_HCM_balance[i], targets_a[0])
            loss_inter += self.inter_loss(targets_a[0], feats_a[i])
            loss_intra += self.intra_loss(targets_a[0], feats_a[i])
 
        loss += NBOD(inputs_balance, factor=self.factor)
        loss += NBOD(inputs_HCM_balance, factor=self.factor_hcm)

        # loss += los_ce + loss_HCM + 0.05*loss_inter + 0.05*0.0125*loss_intra
        loss += los_ce + loss_HCM + 0.05*0.0125*loss_intra # wo inter 

        return loss, los_ce, los_ce, loss_HCM, 0.05*loss_inter + 0.05*0.0125*loss_intra
    
class loss_woCenter(nn.Module):
    def __init__(self, num_class_list, factor, factor_hcm, hcm_num, device):
        super(loss_woCenter, self).__init__()
        print('************ All loss ****************')
        self.bsce_weight = torch.FloatTensor(num_class_list).to(device)
        self.factor = factor
        self.factor_hcm = factor_hcm
        self.hcm_n = hcm_num
        self.intra_loss = Contrastive_Loss_for_Upper_Branch(device)
        self.inter_loss = ContrLoss()

    def get_delay_weight(self, t, w0, wt, T):
        if t < T/3:
            return w0
        elif t >= T/3 and t < (2*T)/3:
            return w0 + (t-(T/3))*(wt*3/T)
        else:
            return wt

    def forward(self, inputs, targets_a, feats_a):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        #w2 = self.get_delay_weight(epoch, 0.01, 0.09, epochs)

        classifier_num = len(inputs)
        loss_HCM = 0
        loss = 0
        los_ce = 0
        loss_inter = 0
        loss_intra = 0

        inputs_HCM_balance = []
        inputs_balance = []
        class_select = inputs[0].scatter(1, targets_a[0].unsqueeze(1), 999999)
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_n]
        mask = torch.zeros_like(inputs[0]).scatter(1, class_select_include_target, 1)
        for i in range(classifier_num):

            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)
            inputs_HCM_balance.append(logits * mask)

            los_ce += F.cross_entropy(logits, targets_a[0])
            loss_HCM += F.cross_entropy(inputs_HCM_balance[i], targets_a[0])
            loss_inter += self.inter_loss(targets_a[0], feats_a[i])
            loss_intra += self.intra_loss(targets_a[0], feats_a[i])

        loss += NBOD(inputs_balance, factor=self.factor)
        loss += NBOD(inputs_HCM_balance, factor=self.factor_hcm)
        # loss += los_ce + loss_HCM + 0.0125*loss_intra
        #loss += los_ce + loss_HCM + 0.05*loss_inter + 0.0125*loss_intra
        loss += los_ce + loss_HCM + 0.05*loss_inter
        # loss += los_ce + loss_HCM

        return loss, los_ce, los_ce, loss_HCM, 0.05*loss_inter + 0.05*0.0125*loss_intra

class loss_woKD(nn.Module):
    def __init__(self, num_class_list, factor, factor_hcm, hcm_num, device):
        super(loss_woKD, self).__init__()
        print('************ All loss ****************')
        self.bsce_weight = torch.FloatTensor(num_class_list).to(device)
        self.factor = factor
        self.factor_hcm = factor_hcm
        self.hcm_n = hcm_num
        self.intra_loss = Contrastive_Loss_for_Upper_Branch(device)
        self.inter_loss = ContrLoss()

    def get_delay_weight(self, t, w0, wt, T):
        if t < T/3:
            return w0
        elif t >= T/3 and t < (2*T)/3:
            return w0 + (t-(T/3))*(wt*3/T)
        else:
            return wt

    def forward(self, inputs, targets_a, feats_a):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        #w2 = self.get_delay_weight(epoch, 0.01, 0.09, epochs)

        classifier_num = len(inputs)
        loss_HCM = 0
        loss = 0
        los_ce = 0
        loss_inter = 0
        loss_intra = 0

        inputs_HCM_balance = []
        inputs_balance = []
        class_select = inputs[0].scatter(1, targets_a[0].unsqueeze(1), 999999)
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_n]
        mask = torch.zeros_like(inputs[0]).scatter(1, class_select_include_target, 1)
        for i in range(classifier_num):

            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)
            inputs_HCM_balance.append(logits * mask)

            los_ce += F.cross_entropy(logits, targets_a[0])
            loss_HCM += F.cross_entropy(inputs_HCM_balance[i], targets_a[0])
            loss_inter += self.inter_loss(targets_a[0], feats_a[i])
            loss_intra += self.intra_loss(targets_a[0], feats_a[i])

        #loss += NBOD(inputs_balance, factor=self.factor)
        #loss += NBOD(inputs_HCM_balance, factor=self.factor_hcm)
        # loss += los_ce + loss_HCM + 0.0125*loss_intra
        #loss += los_ce + loss_HCM + 0.05*loss_inter + 0.0125*loss_intra
        loss += los_ce + loss_HCM + 0.05*loss_inter + 0.05*0.0125*loss_intra
        # loss += los_ce + loss_HCM

        return loss, los_ce, los_ce, loss_HCM, 0.05*loss_inter + 0.05*0.0125*loss_intra

