import os
import sklearn
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import data_process_loader
from torch.utils.data import DataLoader
from model import Oligo
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score,matthews_corrcoef
from metrics import  sensitivity, specificity

from logger import TrainLogger
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)
        return self.avg


def val(model, criterion, dataloader):
	running_loss = AverageMeter()
	pred_list = []
	pred_cls_list = []
	label_list = []
	_efficacy = []
	_label = []
	for i, data in enumerate(dataloader):
		siRNA = data[0].to(device)
		mRNA = data[1].to(device)
		siRNA_FM = data[2].to(device)
		mRNA_FM = data[3].to(device)
		label = data[4].to(device)
		td = data[6].to(device)
		pred,_,_ = model(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
		_efficacy += list(pred[:,1].detach().cpu().numpy())
		_label += list(label.float().detach().cpu().numpy())
		loss = criterion(pred[:,1],label.float()) 
		label = np.array([int(i > 0.7) for i in label])
		pred_cls = torch.argmax(pred, dim=-1)
		pred_prob = F.softmax(pred, dim=-1) # pred_prob = pred #
		pred_prob, indices = torch.max(pred_prob, dim=-1)
		pred_prob[indices == 0] = 1. - pred_prob[indices == 0]
		pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
		pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
		label_list.append(label)
		running_loss.update(loss, label.shape[0])
	pred = np.concatenate(pred_list, axis=0)
	pred_cls = np.concatenate(pred_cls_list, axis=0)
	label = np.concatenate(label_list, axis=0)
	acc = accuracy_score(label, pred_cls)
	sen = sensitivity(label,pred_cls)
	spe = specificity(label,pred_cls)
	pre = precision_score(label, pred_cls)
	rec = recall_score(label, pred_cls)
	f1score=f1_score(label,pred_cls)
	rocauc = roc_auc_score(label, pred)
	prauc=average_precision_score(label, pred)
	mcc=matthews_corrcoef(label,pred_cls)
	epoch_loss = running_loss.get_average()
	running_loss.reset()
	return epoch_loss, acc, sen, spe, pre, rec, f1score, rocauc, prauc, mcc, label, pred, _efficacy, _label

def encode_sequence(seq):
    base_dict = {'A':[1,0,0,0],'U':[0,1,0,0],'C':[0,0,1,0],'G':[0,0,0,1]}
    seq_encoded = []
    for base in seq:
        seq_encoded.append(base_dict[base])
    
    # flatten list
    seq_encoded =   [item for items in seq_encoded for item in items]
    return seq_encoded

def get_prediction_class(x):
    if x['label'] == '1': # is actually positive
        if x['pred'] == 1: # correct prediction
            return "TP"
        else: # incorrectly predicted to be negative
            return "FN" 
    else: # is actually negative
        if x['pred'] == 1: # incorrectly predicted to be positive
            return "FP"
        else: # correctly predicted to be negative
            return "TN"

def test(Args):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(Args.cuda)
	random.seed(Args.seed)
	os.environ['PYTHONHASHSEED']=str(Args.seed)
	np.random.seed(Args.seed)
	train_df = pd.read_csv(Args.path + Args.datasets[0] + '.csv', dtype=str)
	valid_df = pd.read_csv(Args.path + Args.datasets[1] + '.csv',  dtype=str)
	test_df = pd.read_csv(Args.path + Args.datasets[1] + '.csv', dtype=str)
	params = {'batch_size': Args.batch_size,
			'shuffle': False,
			'num_workers': 0,
			'drop_last': False}
	train_ds = DataLoader(data_process_loader(train_df.index.values, train_df.label.values,train_df.y.values, train_df, Args.datasets[0],Args.path), **params)
	valid_ds = DataLoader(data_process_loader(valid_df.index.values, valid_df.label.values,valid_df.y.values, valid_df, Args.datasets[1],Args.path),**params)
	test_ds = DataLoader(data_process_loader(test_df.index.values, test_df.label.values,test_df.y.values, test_df, Args.datasets[1],Args.path), **params)
	OFmodel = Oligo(vocab_size = Args.vocab_size, embedding_dim = Args.embedding_dim, lstm_dim = Args.lstm_dim,  n_head = Args.n_head, n_layers = Args.n_layers, lm1 = Args.lm1, lm2 = Args.lm2).to(device)
	OFmodel.load_state_dict(torch.load(Args.best_model,map_location=device))
	criterion = nn.MSELoss()
	params = dict(
		data_path=Args.path,
		save_dir=Args.output_dir,
		dataset=Args.datasets[0],
		batch_size=Args.batch_size
	)
	logger = TrainLogger(params)
	logger.info(f"Number of train: {train_df.shape[0]}")
	logger.info(f"Number of val: {valid_df.shape[0]}")
	logger.info(f"Number of test: {test_df.shape[0]}")
	print('-----------------Start testing!-----------------')
	val_loss, val_acc, val_sen, val_spe, val_pre, val_rec, val_f1, val_rocauc, val_prauc, val_mcc, val_label, val_pred,val_efficacy, val_label2 = val(OFmodel, criterion, valid_ds)
	msg = "val_acc-%.4f,val_f1-%.4f, val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_loss-%.4f ***" % (val_acc,val_f1,val_pre, val_rec,val_rocauc,val_prauc,val_loss)
	logger.info(msg)
