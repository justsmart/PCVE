from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import math,random
def batch_perm(batch_data,masks):
    n = batch_data.shape[1]
    rotated_flat_matrices =torch.zeros_like(batch_data)
    for i, data in enumerate(batch_data):
        rotation_list = generate_rotation_list(n)
        index_row = create_index_matrix(n,rotation_list,masks[i])
        index_col = torch.arange(n).view(1, -1).expand(n, n)
        new_data = data[index_row,index_col]
        rotated_flat_matrices[i] = new_data
    return rotated_flat_matrices
def gen_rotat_index(N, M, masks):
    index_rows = []
    for i in range(N):
        rotation_list = generate_rotation_list(M)
        index_row = create_index_matrix(M, rotation_list, masks[i])
        index_rows.append(index_row)
    return index_rows
def generate_rotation_list(n):
    """生成一个从 0 到 N-1 的随机排序列表"""
    rotation_list = np.arange(n)
    np.random.shuffle(rotation_list)
    return rotation_list

def create_index_matrix(n, rotation_list, mask):
    """创建一个索引矩阵，用于重构旋转后的矩阵"""
    index_row = torch.arange(n).view(-1,1).repeat(1,n) #[[0,0,0,0] [1,1,1,1] [2,2,2,2] [3,3,3,3]]
    for col in range(n):
        rotation_amount = rotation_list[col]
        for i in range(n):
            step = rotation_amount
            if mask[i] == 1:  # 只考虑未被屏蔽的行
                new_position = i
                while step>0:
                    new_position = (new_position + 1) % n
                    step = step - 1 if mask[new_position] != 0 else step
                index_row[new_position, col] = i
    return index_row
def loadMvMlDataFromMat(mat_path):
    data = scipy.io.loadmat(mat_path)
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    if labels.min() == -1:
        labels = (labels + 1) * 0.5
    if labels.shape[0] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    elif labels.shape[1] in mv_data[0].shape:
        total_sample_num = labels.shape[1]
    else:
        raise ValueError("Label dimensions do not match the number of samples")
    if total_sample_num != mv_data[0].shape[0]:
        mv_data = [v_data.T for v_data in mv_data]
    if total_sample_num != labels.shape[0]:
        labels = labels.T
    assert mv_data[0].shape[0]==labels.shape[0]==total_sample_num

    mv_data = [StandardScaler().fit_transform(v_data.astype(np.float32)) for v_data in mv_data]
    random.seed(1)
    rand_index=list(range(total_sample_num))
    random.shuffle(rand_index)
    return [v_data[rand_index] for v_data in mv_data],labels[rand_index],total_sample_num

def loadMfDIMvMlDataFromMat(mat_path, fold_mat_path,fold_idx=0):
    data = scipy.io.loadmat(mat_path)
    datafold = scipy.io.loadmat(fold_mat_path)
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    if labels.min() == -1:
            labels = (labels + 1) * 0.5
    if labels.shape[0] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    elif labels.shape[1] in mv_data[0].shape:
        total_sample_num = labels.shape[1]
    else:
        raise ValueError("Label dimensions do not match the number of samples")
    if total_sample_num != mv_data[0].shape[0]:
        mv_data = [v_data.T for v_data in mv_data]
    if total_sample_num != labels.shape[0]:
        labels = labels.T
    assert mv_data[0].shape[0]==labels.shape[0]==total_sample_num



    folds_data = datafold['folds_data']
    folds_label = datafold['folds_label']
    folds_sample_index = datafold['folds_sample_index']
    inc_view_indicator = np.array(folds_data[0, fold_idx], 'int32')
    inc_label_indicator = np.array(folds_label[0, fold_idx], 'int32')  # incomplete label index
    sample_index = np.array(folds_sample_index[0, fold_idx], 'int32').reshape(-1)-1 # index start from 0
    labels,inc_view_indicator,inc_label_indicator = labels[sample_index],inc_view_indicator[sample_index],inc_label_indicator[sample_index]
    mv_data = [v_data[sample_index,:] for v,v_data in enumerate(mv_data)]

    assert inc_view_indicator.shape[0]==inc_label_indicator.shape[0]==sample_index.shape[0]==labels.shape[0]
    inc_mv_data = [(StandardScaler().fit_transform(v_data.astype(np.float32))*inc_view_indicator[:,v:v+1]) for v,v_data in enumerate(mv_data)]
    inc_labels = labels*inc_label_indicator

    return inc_mv_data,inc_labels,labels,inc_view_indicator,inc_label_indicator,total_sample_num

class ComDataset(Dataset):
    def __init__(self,mat_path,training_ratio=0.7,val_ratio=0.15,mode='train',semisup=False):
        self.mv_data, self.labels, self.total_sample_num= loadMvMlDataFromMat(mat_path)
        self.train_sample_num = math.ceil(self.total_sample_num * training_ratio)
        self.val_sample_num = math.ceil(self.total_sample_num * val_ratio)
        self.test_sample_num = self.total_sample_num - self.train_sample_num - self.val_sample_num
        if mode=='train':
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in self.mv_data]
            self.cur_labels = self.labels[:self.train_sample_num]
        elif mode=='val':
            self.cur_mv_data = [v_data[self.train_sample_num:self.train_sample_num+self.val_sample_num] for v_data in self.mv_data]
            self.cur_labels = self.labels[self.train_sample_num:self.train_sample_num+self.val_sample_num]
        else:
            self.cur_mv_data = [v_data[self.train_sample_num+self.val_sample_num:] for v_data in self.mv_data]
            self.cur_labels = self.labels[self.train_sample_num+self.val_sample_num:]
        self.mode = mode
        self.classes_num = self.labels.shape[1]
        self.d_list = [da.shape[1] for da in self.mv_data]
    def __len__(self):
        if self.mode == 'train':
            return self.train_sample_num
        elif self.mode == 'val':
            return self.val_sample_num
        else: return self.test_sample_num

    def __getitem__(self, index):
        data = [torch.tensor(v[index],dtype=torch.float) for v in self.cur_mv_data]
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        return data,label, data, label

class IncDataset(Dataset):
    def __init__(self,mat_path, fold_mat_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',semisup=False):
        inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num= loadMfDIMvMlDataFromMat(mat_path,fold_mat_path,fold_idx)
        row_indexs = torch.stack(gen_rotat_index(total_sample_num, inc_V_ind.shape[1], inc_V_ind)).numpy()


        self.train_sample_num = math.ceil(total_sample_num * training_ratio)
        self.val_sample_num = math.ceil(total_sample_num * val_ratio)
        self.test_sample_num = total_sample_num - self.train_sample_num - self.val_sample_num
        if mode=='train':
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in inc_mv_data]
            self.cur_labels = inc_labels[:self.train_sample_num]
            self.cur_inc_V_ind = inc_V_ind[:self.train_sample_num]
            self.cur_inc_L_ind = inc_L_ind[:self.train_sample_num]
            self.cur_row_indexs = row_indexs[:self.train_sample_num]
        elif mode=='val':
            self.cur_mv_data = [v_data[self.train_sample_num:self.train_sample_num+self.val_sample_num] for v_data in inc_mv_data]
            self.cur_labels = labels[self.train_sample_num:self.train_sample_num+self.val_sample_num]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num:self.train_sample_num+self.val_sample_num]
            self.cur_inc_L_ind = np.ones_like(inc_L_ind[self.train_sample_num:self.train_sample_num+self.val_sample_num])
            self.cur_row_indexs = row_indexs[self.train_sample_num:self.train_sample_num+self.val_sample_num]
        else:
            self.cur_mv_data = [v_data[self.train_sample_num+self.val_sample_num:] for v_data in inc_mv_data]
            self.cur_labels = labels[self.train_sample_num+self.val_sample_num:]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num+self.val_sample_num:]
            self.cur_inc_L_ind = np.ones_like(inc_L_ind[self.train_sample_num+self.val_sample_num:])
            self.cur_row_indexs = row_indexs[self.train_sample_num+self.val_sample_num:]

        self.mode = mode
        self.classes_num = labels.shape[1]
        self.d_list = [da.shape[1] for da in inc_mv_data]

    def __len__(self):
        if self.mode == 'train':
            return self.train_sample_num
        elif self.mode == 'val':
            return self.val_sample_num
        else: return self.test_sample_num

    def __getitem__(self, index):
        data = [torch.tensor(v[index],dtype=torch.float) for v in self.cur_mv_data]
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        inc_V_ind = torch.tensor(self.cur_inc_V_ind[index], dtype=torch.int32)
        inc_L_ind = torch.tensor(self.cur_inc_L_ind[index], dtype=torch.int32)
        row_index = torch.tensor(self.cur_row_indexs[index],dtype=torch.int32)
        return data,label,inc_V_ind,inc_L_ind,row_index

def getComDataloader(matdata_path,training_ratio=0.7,val_ratio=0.15,mode='train',batch_size=1,num_workers=1,shuffle=False):
    dataset = ComDataset(matdata_path, training_ratio=training_ratio, val_ratio=val_ratio, mode=mode)
    dataloder = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return dataloder,dataset

def getIncDataloader(matdata_path, fold_matdata_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',batch_size=1,num_workers=1,shuffle=False):
    dataset = IncDataset(matdata_path, fold_matdata_path, training_ratio=training_ratio, val_ratio=val_ratio, mode=mode, fold_idx=fold_idx)
    dataloder = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return dataloder,dataset
