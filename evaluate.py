import numpy as np
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
class tsne_function(object):
    def __init__(self, num_classes, id2cat):
        self.tsne_img = []
        self.tsne_label = []
        self.num_classes = num_classes
        self.id2cat = id2cat
        #print(id2cat)
    def store_data(self, img, label):
        self.tsne_img.append(img.detach().cpu())
        self.tsne_label.append(label.detach().cpu())
    
    def visualize(self, output_path):
        # print('Total Img shape: {}, Label shape: {}'.format(np.array(self.tsne_img).shape, np.array(self.tsne_label).shape))
        g = [23,24,25,26]
        img = np.concatenate(self.tsne_img)
        label = np.concatenate(self.tsne_label)
        t0 = time.time()
        
        tsne = TSNE(n_components=2, init='pca').fit_transform(img)
        # print(label)
        #colors_dic = mpl.colors.cnames
        #colors = list(colors_dic.values())
        #colors = colors[10::]
        colors = ['black', 'red', 'orangered', 'darkgoldenrod', 'olivedrab', 'lime', 
                    'cyan', 'blue', 'deeppink', 'purple', 'slategray','gold' ]

        # print(colors[0])
        # markers = ["o", "v", "^", "<", ">", "1", "4", "8", "s", "p", "*", "h", "x", "D"]
        markers = ["o", "v", "^", "<", ">", "x", "D"]
        #colors = {23:'red', 25:'blue', 26:'black', 24:'yellow'}
        #colors = ['red', 'yellow', 'blue', 'black']
        
        fig = plt.figure(figsize=(16,16))
        for cls in range(self.num_classes):
            #print(cls)
            if cls in g:
                idx = np.where(label==cls)
                idx = idx[0] # 原先idx=(array([ 3, 10, 21, 27, 44, 52]), dtype)
                        # print(colors[cls])
                        #print(cls)
                    
                        #plt.scatter(tsne[idx, 0], tsne[idx, 1], label=self.id2cat[str(cls)], c=colors[cls])
                plt.scatter(tsne[idx, 0], tsne[idx, 1], label=self.id2cat[str(cls)], c=colors[cls%len(colors)], marker=markers[cls%len(markers)])

        
        title = 'Tsne of final fc output_time={}'.format(round(time.time()-t0, 3))
        plt.title(title)
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.legend(loc='best')
        output_path = os.path.join(output_path, '{}.png'.format(title))
        # plt.savefig(output_path)
        #plt.savefig('./Tsne of head shot.png')
        plt.savefig('./Tsne of feat_train.png')
        plt.close()


class FusionMatrix(object):
    def __init__(self, num_classes, id2cat, order_keys=None):
        self.num_classes = num_classes
        self.reset()
        self.id2cat = id2cat
        self.order_keys = order_keys
        if self.order_keys != None:
            self.new_label = {order_keys[i]: i for i in range(len(order_keys))} # {new key: 0...}
            print('********', order_keys)
            print(self.new_label)
        print(id2cat)
        
    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def trans_label(self, input_arr):
        new_arr = []
        for i in range(len(input_arr)):
            old_key = self.id2cat[str(input_arr[i])]
            trans_label = self.new_label[old_key]
            new_arr.append(trans_label)
        
        return np.array(new_arr)


    def update(self, output, label):
        if self.order_keys != None: # 排序沒照學姊的setting
            output = self.trans_label(output)
            label = self.trans_label(label)
        length = output.shape[0]
        for i in range(length):
            self.matrix[output[i], label[i]] += 1
    
    def plot_bar_pre(self, cfg, baseline=False):

        pre = self.get_pre_per_class()
        title_pre = 'Precision for per class, mean={}'.format(round(np.mean(pre), 2))
        print('Precision: ', np.mean(pre))
        xaxis = [i for i in range(len(pre))]

        plt.bar(xaxis, pre)
        for i in range(len(pre)):
            plt.text(i, pre[i], round(pre[i],2), ha='center', fontsize=7)
        plt.xticks(rotation=90)
        plt.xlim(-1,len(pre))
        plt.xlabel('Classes')
        plt.ylabel('Precision')
        plt.title(title_pre)
        ax = plt.gca()
        x_major = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major)
        if baseline == False:
            plt.savefig('{}/{}/{}.png'.format(cfg.OUTPUT_DIR, cfg.NAME, title_pre))
        else:
            plt.savefig('{}/{}/Baseline_{}.png'.format(cfg.OUTPUT_DIR, cfg.NAME, title_pre))
        plt.close()
    
    def plot_bar_rec(self, output_path, LT_group):

        rec = self.get_rec_per_class()
        title_rec = 'Recall for per class, mean={}'.format(round(np.mean(rec), 2))
        print('Recall: ', np.mean(rec))
        xaxis = [val for key, val in self.id2cat.items()]
        ########## produce group
        new_xaxis = [[], [], [], []]
        new_rec = [[], [], [], []]
        colors = [[], [], [], []]
        for i in range(len(rec)):
            if xaxis[i] in LT_group['Head']:
                new_xaxis[0].append(xaxis[i])
                new_rec[0].append(rec[i])  
                colors[0].append('Red')  
            elif xaxis[i] in LT_group['Many']:
                new_xaxis[1].append(xaxis[i])
                new_rec[1].append(rec[i]) 
                colors[1].append('Green')  
            elif xaxis[i] in LT_group['Medium']:
                new_xaxis[2].append(xaxis[i])
                new_rec[2].append(rec[i])   
                colors[2].append('Blue') 
            else:
                new_xaxis[3].append(xaxis[i])
                new_rec[3].append(rec[i]) 
                colors[3].append('silver') 
  
        new_xaxis = np.concatenate(new_xaxis, axis=0)
        new_rec = np.concatenate(new_rec, axis=0)
        colors = np.concatenate(colors, axis=0)
        ##########

        plt.figure(figsize=(15,10))
        plt.bar(new_xaxis, new_rec, color=colors)
        for i in range(len(new_rec)):
            plt.text(i, new_rec[i], round(new_rec[i],2), ha='center', fontsize=7)
        plt.xticks(rotation=90)
        plt.xlim(-1,len(new_rec))
        plt.xlabel('Classes')
        plt.ylabel('Recall')
        plt.title(title_rec)
        ax = plt.gca()
        x_major = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major)
       
        #plt.show()
        plt.savefig('{}/{}.png'.format(output_path, title_rec))
        plt.close()
  
    def get_rec_per_class(self): # recall
        rec = np.array(
            [
                self.matrix[i, i] / self.matrix[:, i].sum()
                for i in range(self.num_classes)
            ]
        )
        rec[np.isnan(rec)] = 0
        
        return rec

    def get_pre_per_class(self): # precision 
        pre = np.array(
            [
                self.matrix[i, i] / self.matrix[i, :].sum()
                for i in range(self.num_classes)
            ]
        )
        pre[np.isnan(pre)] = 0
        
        return pre

    def get_accuracy(self):
        acc = (
            np.sum([self.matrix[i, i] for i in range(self.num_classes)])
            / self.matrix.sum()
        )
        return acc

    def plot_confusion_matrix(self, normalize = False, acc=None, cmap=plt.cm.Blues, output_path=None):
        # Compute confusion matrix
        cm = self.matrix.T
        recall_per_block = self.cal_rec_per_block(cm)

        title = 'Confusion matrix_cls_acc={}'.format(np.round(acc,3))
        matrix = recall_per_block
        matrix_num = cm

        fig, ax = plt.subplots(figsize=(20,20))
        im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        if self.order_keys != None:
            x_y_label = [key for key, val in self.new_label.items()]
        else:
            x_y_label = [val for key, val in self.id2cat.items()]
        ax.set(xticks=np.arange(matrix.shape[1]), # 排row, col有幾個
               yticks=np.arange(matrix.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=x_y_label, yticklabels=x_y_label,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                block_text = '{}\n{}%'.format(matrix_num[i, j], matrix[i,j])
                #print(block_text)
                ax.text(j, i, block_text,
                        ha="center", va="center",
                        color="white" if matrix[i, j] > thresh else "black")
        fig.tight_layout()
        outpt_path = os.path.join(output_path, '{}.png'.format(title))
        plt.savefig(outpt_path)
        
        plt.close()
        # return fig
    def cal_rec_per_block(self, matrix):
        new_matrix = np.zeros_like(matrix)
        #print(new_matrix)
        #print(matrix)
        for i in range(matrix.shape[0]):
            num_per_row = matrix[i, :].sum()
            for j in range(matrix.shape[1]):
                # print(i,j)
                new_matrix[i, j] = (matrix[i,j]/num_per_row)*100
        
        return new_matrix


def accuracy(output, label):
    cnt = label.shape[0]
    true_count = (output == label).sum()
    now_accuracy = true_count / cnt
    return now_accuracy, cnt
