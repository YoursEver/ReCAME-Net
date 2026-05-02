import os
import torch
import datetime
import numpy as np
import time
import logging
# use GPU
import csv
from tqdm import tqdm
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference_wo_label(net, loader, path):
    id2cat = {0: "01_Hm", 1: "01_Se", 2: "01_Su", 3: "01_Sw", 4: "01_Tm", 
              5: "02_As", 6: "02_De", 7: "02_Em", 8: "02_Gen", 9: "02_Hsp", 
              10: "02_Sg", 11: "02_Sl", 12: "02_Sz", 13: "03_Im", 14: "04_Ms", 
              15: "05_Gm", 16: "05_Pa", 17: "05_Pb", 18: "05_Sinsi", 19: "05_Sinter", 
              20: "05_Sr", 21: "07_Rg", 22: "09_Ar", 23: "11_Eh", 24: "11_Ej", 
              25: "11_Ep", 26: "11_Si", 27: "11_Tc", 28: "11_Td", 29: "11_Ts", 30: "12_Ema"}

  
    #label_test_number = []
    
    filename_arr = []
    predict_p, predict_cls = [], [] # avg
    predict_p_0, predict_p_1 = [], [] # expert p
    predict_cls_0, predict_cls_1 = [], [] # expert predict cls
    bs = 16
    fn = nn.Softmax(dim=1)

    with torch.no_grad():
        time_start = time.time()  
        net.eval()
        for batch_idx, (inputs, _, _, _, filename) in enumerate(tqdm(loader)): 
            inputs = inputs.to(device)
            
            #print(bs)
            images = []
            for k in range(2):
                images.append(inputs)
              
            outputs, feature = net(images)
            
            for j in range(len(outputs)):
                outputs[j] = fn(outputs[j])

            p0, pred_cls0 = outputs[0].max(1)
            p1, pred_cls1 = outputs[1].max(1)
            
            avg_logit = torch.mean(torch.stack(outputs), dim=0)
            p, pred_cls = avg_logit.max(1)

            
            filename_arr.append(filename)
            #
            # predict_arr.append(predicted.detach().cpu())
            
            predict_p.append(p.detach().cpu())
            predict_p_0.append(p0.detach().cpu())
            predict_p_1.append(p1.detach().cpu())
            predict_cls.append(pred_cls.detach().cpu())
            predict_cls_0.append(pred_cls0.detach().cpu())
            predict_cls_1.append(pred_cls1.detach().cpu())

            # if batch_idx == 1:
            #     break
        
        #print(predict_cls)
        time_end = time.time()  
        cost = time_end-time_start
        print('Total inferenece time: {}'.format(cost))

        filename_arr = np.array(filename_arr)
        #predict_arr = np.array(predict_arr)
        #print(predict_p)
        predict_p = np.array(predict_p)
        predict_p_0 = np.array(predict_p_0)
        predict_p_1 = np.array(predict_p_1)
        predict_cls = np.array(predict_cls)
        predict_cls_0 = np.array(predict_cls_0)
        predict_cls_1 = np.array(predict_cls_1)

        cat_filename_arr = np.concatenate(filename_arr, axis=0)
        # cat_predict_arr = np.concatenate(predict_arr, axis=0)
        cat_predict_cls = np.concatenate(predict_cls, axis=0)
        cat_predict_cls_0 = np.concatenate(predict_cls_0, axis=0)
        cat_predict_cls_1 = np.concatenate(predict_cls_1, axis=0)
        cat_predict_p = np.concatenate(predict_p, axis=0)
        cat_predict_p_0 = np.concatenate(predict_p_0, axis=0)
        cat_predict_p_1 = np.concatenate(predict_p_1, axis=0)

        new_cat_predict_cls = np.vectorize(id2cat.get)(cat_predict_cls)
        new_cat_predict_cls_0 = np.vectorize(id2cat.get)(cat_predict_cls_0)
        new_cat_predict_cls_1 = np.vectorize(id2cat.get)(cat_predict_cls_1)
        #print'

        ex_cat_filename_arr = np.expand_dims(cat_filename_arr, axis=1)
        ex_new_cat_predict_cls = np.expand_dims(new_cat_predict_cls, axis=1)
        ex_new_cat_predict_cls_0 = np.expand_dims(new_cat_predict_cls_0, axis=1)
        ex_new_cat_predict_cls_1 = np.expand_dims(new_cat_predict_cls_1, axis=1)
        ex_cat_predict_p = np.expand_dims(cat_predict_p, axis=1)
        ex_cat_predict_p_0 = np.expand_dims(cat_predict_p_0, axis=1)
        ex_cat_predict_p_1 = np.expand_dims(cat_predict_p_1, axis=1)

        
        # arr = np.concatenate((ex_cat_filename_arr, ex_new_cat_predict_cls_0, ex_new_cat_predict_cls_1,
        #                       ex_new_cat_predict_cls, ex_cat_predict_p, ex_cat_predict_p_0, ex_cat_predict_p_1), axis=1)
        
        arr = np.concatenate((ex_cat_filename_arr, ex_cat_predict_p_0, ex_cat_predict_p_1, ex_cat_predict_p, 
                              ex_new_cat_predict_cls_0, ex_new_cat_predict_cls_1, ex_new_cat_predict_cls), axis=1)
        
        
        #print(cat_filename_arr, cat_predict_arr)
        #print(arr)
    
    csv_output_path = os.path.join(path, 'UMC_predict_LT.csv')

    bs = np.expand_dims(np.array([bs]), axis=1)
    cost = np.expand_dims(np.array([int(cost)]), axis=1)
    bs_cost = np.concatenate((bs, cost), axis=1)
    print(bs_cost)
    first_rol = np.array(['Batch size', 'Cost time(sec)'])
    first_rol = np.expand_dims(first_rol, axis=0)
    second_rol = np.array(['Filename', 'Expert0 p', 'Expert1 p', 'Avg p', 'Expert0 cls', 'Expert1 cls', 'Avg cls'])
    second_rol = np.expand_dims(second_rol, axis=0)
    with open (csv_output_path, 'w') as f:
        writer = csv.writer(f, delimiter=',')

        #### write bs cost
        writer.writerows(first_rol)
        writer.writerows(bs_cost)
        ##### write predict
        writer.writerows(second_rol)
        writer.writerows(arr)
        f.close()

def inference_multi_expert(net, loader, path, num_classifier=2):
    id2cat = {0: "01_Hm", 1: "01_Se", 2: "01_Su", 3: "01_Sw", 4: "01_Tm", 
              5: "02_As", 6: "02_De", 7: "02_Em", 8: "02_Gen", 9: "02_Hsp", 
              10: "02_Sg", 11: "02_Sl", 12: "02_Sz", 13: "03_Im", 14: "04_Ms", 
              15: "05_Gm", 16: "05_Pa", 17: "05_Pb", 18: "05_Sinsi", 19: "05_Sinter", 
              20: "05_Sr", 21: "07_Rg", 22: "09_Ar", 23: "11_Eh", 24: "11_Ej", 
              25: "11_Ep", 26: "11_Si", 27: "11_Tc", 28: "11_Td", 29: "11_Ts", 30: "12_Ema"}

  
    #label_test_number = []
    
    filename_arr = []
    predict_p, predict_cls = [[] for i in range(num_classifier+1)], [[] for i in range(num_classifier+1)]
    bs = 16
    fn = nn.Softmax(dim=1)

    with torch.no_grad():
        time_start = time.time()  
        net.eval()
        for batch_idx, (inputs, _, _, _, filename) in enumerate(tqdm(loader)): 
            inputs = inputs.to(device)
            
            #print(bs)
            images = []
            for k in range(2):
                images.append(inputs)
              
            outputs, feature = net(images)
            
            for j in range(len(outputs)):
                outputs[j] = fn(outputs[j]) # p
                p, pred_cls = outputs[j].max(1) # p與class
                predict_p[j].append(p.detach().cpu())
                predict_cls[j].append(pred_cls.detach().cpu())


            avg_logit = torch.mean(torch.stack(outputs), dim=0)
            p, pred_cls = avg_logit.max(1)

            filename_arr.append(filename)
            predict_p[-1].append(p.detach().cpu())
            predict_cls[-1].append(pred_cls.detach().cpu())

            # if batch_idx == 1:
            #     break
        
        #print(predict_cls)
        time_end = time.time()  
        cost = time_end-time_start
        print('Total inferenece time: {}'.format(cost))

        filename_arr = np.array(filename_arr)
        #predict_arr = np.array(predict_arr)
        #print(predict_p)
        new_cat_predict_cls = [[] for i in range(num_classifier+1)]
        for j in range(num_classifier+1):
            predict_p[j] = np.array(predict_p[j])
            predict_cls[j] = np.array(predict_cls[j])
            predict_p[j] = np.concatenate(predict_p[j], axis=0) # 
            predict_cls[j] = np.concatenate(predict_cls[j], axis=0)
            new_cat_predict_cls[j] = np.vectorize(id2cat.get)(predict_cls[j]) # 

            predict_p[j] = np.expand_dims(predict_p[j], axis=1)
            new_cat_predict_cls[j] = np.expand_dims(new_cat_predict_cls[j], axis=1)

        cat_filename_arr = np.concatenate(filename_arr, axis=0)
        # cat_predict_arr = np.concatenate(predict_arr, axis=0)

        #print'

        ex_cat_filename_arr = np.expand_dims(cat_filename_arr, axis=1)

        arr = np.concatenate((ex_cat_filename_arr, predict_p[0]), axis=1)
        for j in range(1, num_classifier+1):
            arr = np.concatenate((arr, predict_p[j]), axis=1)
        
        for j in range(num_classifier+1):
            arr = np.concatenate((arr, new_cat_predict_cls[j]), axis=1)

    csv_output_path = os.path.join(path, 'UMC_predict_LT.csv')

    bs = np.expand_dims(np.array([bs]), axis=1)
    cost = np.expand_dims(np.array([int(cost)]), axis=1)
    bs_cost = np.concatenate((bs, cost), axis=1)
    print(bs_cost)
    first_rol = np.array(['Batch size', 'Cost time(sec)'])
    first_rol = np.expand_dims(first_rol, axis=0)
    second_rol = np.array(['Filename', 'Expert0 p', 'Expert1 p', 'Avg p', 'Expert0 cls', 'Expert1 cls', 'Avg cls']) # 多expert rol在這改
    second_rol = np.expand_dims(second_rol, axis=0)
    with open (csv_output_path, 'w') as f:
        writer = csv.writer(f, delimiter=',')

        #### write bs cost
        writer.writerows(first_rol)
        writer.writerows(bs_cost)
        ##### write predict
        writer.writerows(second_rol)
        writer.writerows(arr)
        f.close()





