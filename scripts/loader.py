import pandas as pd
import numpy as np
import random
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import torch

nucleic_char = ['A', 'U', 'C', 'G', 'X'] # 
structure_char = ['.', '(', ')']

# ACGU 1234

enc_protein = OneHotEncoder().fit(np.array(nucleic_char).reshape(-1, 1))
enc_structure =  LabelEncoder().fit(np.array(structure_char))
def sequence_OneHot(x):
    return enc_protein.transform(np.array(x).reshape(-1, 1)).toarray().T

def structure_Label(x):
    return enc_structure.transform(np.array(x).reshape(-1, 1)).T

def Embed(RNA):
    nucleotides = 'ACGU' 
    char_to_int = dict((c, i + 1 ) for i, c in enumerate(nucleotides))
    return [char_to_int[i] for i in RNA]


class data_process_loader(data.Dataset):
    def __init__(self, df_index, labels, y, df, dataset,root):
        'Initialization'
        self.labels = labels
        self.df_index = df_index
        self.y = y
        self.df = df
        self.dataset = dataset
        self.siRNA_ref = {}
        self.root = root
        with open(root + '/fasta/' + self.dataset + '_siRNA.fa') as f:
            for line in f:
                if line.startswith('>'):
                    name = line.strip().replace('>','')
                else:
                    seq = line.strip()
                    self.siRNA_ref[seq] = name
        self.mRNA_ref = {}
        with open(root + '/fasta/' + self.dataset + '_mRNA.fa') as f:
            for line in f:
                if line.startswith('>'):
                    name = line.strip().replace('>','')
                else:
                    seq = line.strip()
                    self.mRNA_ref[seq] = name
    def __len__(self):
        return len(self.df_index)
    def __getitem__(self, index):
        index = self.df_index[index]
        label = float(self.labels[index])
        y = np.int64(self.y[index])
        # siRNA
        siRNA_seq = self.df.iloc[index]['siRNA']
        siRNA_seq = [*siRNA_seq]
        siRNA_seq = sequence_OneHot(siRNA_seq)
        siRNA = np.expand_dims(siRNA_seq, axis=2).transpose([2, 1, 0])
        # mRNA
        mRNA_seq = self.df.iloc[index]['mRNA']
        mRNA_seq = [*mRNA_seq]
        mRNA_seq = sequence_OneHot(mRNA_seq)
        mRNA = np.expand_dims(mRNA_seq, axis=2).transpose([2, 1, 0])
        # siRNA RNA-FM
        siRNA_seq = self.df.iloc[index]['siRNA']
        siRNA_FM = np.load(self.root + '/RNAFM/' + self.dataset + '_siRNA/representations/'+ str(self.siRNA_ref[siRNA_seq]) + '.npy')
        # mRNA RNA-FM
        mRNA_seq = self.df.iloc[index]['mRNA']
        mRNA_FM = np.load(self.root + '/RNAFM/' + self.dataset + '_mRNA/representations/' + str(self.mRNA_ref[mRNA_seq])+'.npy') 
        td = self.df.iloc[index]['td']
        td = torch.tensor([float(i) for i in td.split(',')])
        return  siRNA, mRNA, siRNA_FM, mRNA_FM, label, y, td




class data_process_loader_infer(data.Dataset):
    def __init__(self, df_index, df, dataset):
        self.df_index = df_index
        self.df = df
        self.dataset = dataset
        self.siRNA_ref = {}
        with open('./data/infer/' + self.dataset + '/siRNA.fa') as f:
            for line in f:
                if line.startswith('>'):
                    name = line.strip().replace('>','')
                else:
                    seq = line.strip()
                    self.siRNA_ref[seq] = name
        self.mRNA_ref = {}
        with open('./data/infer/' + self.dataset + '/mRNA.fa') as f:
            for line in f:
                if line.startswith('>'):
                    name = line.strip().replace('>','')
                else:
                    seq = line.strip()
                    self.mRNA_ref[seq] = name
    def __len__(self):
        return len(self.df_index)
    def __getitem__(self, index):
        index = self.df_index[index]
        # siRNA
        siRNA_seq = self.df.iloc[index]['siRNA']
        siRNA_seq = [*siRNA_seq]
        siRNA_seq = sequence_OneHot(siRNA_seq)
        siRNA = np.expand_dims(siRNA_seq, axis=2).transpose([2, 1, 0])
        # mRNA
        mRNA_seq = self.df.iloc[index]['mRNA']
        mRNA_seq = [*mRNA_seq]
        mRNA_seq = sequence_OneHot(mRNA_seq)
        mRNA = np.expand_dims(mRNA_seq, axis=2).transpose([2, 1, 0])
        # siRNA RNA-FM
        siRNA_seq = self.df.iloc[index]['siRNA']
        siRNA_FM = np.load('./data/infer/' + self.dataset + '/siRNA/representations/'+ str(self.siRNA_ref[siRNA_seq]) + '.npy')
        # mRNA RNA-FM
        mRNA_seq = self.df.iloc[index]['mRNA']
        mRNA_FM = np.load('./data/infer/' + self.dataset + '/mRNA/representations/' + str(self.mRNA_ref[mRNA_seq])+'.npy') 
        td = self.df.iloc[index]['td']
        td = torch.tensor(td).to(torch.float32)
        return  siRNA, mRNA, siRNA_FM, mRNA_FM, td