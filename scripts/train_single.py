import os
import sklearn
import random
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import data_process_loader
from torch.utils.data import DataLoader
from model import Oligo
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score,roc_curve,auc,precision_recall_curve,matthews_corrcoef
from metrics import  sensitivity, specificity
from sklearn.utils import shuffle
from logger import TrainLogger
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def get_kfold_data_2(i, datasets, k=5, v=1):
    datasets = shuffle_dataset(datasets, 42).reset_index(drop=True)
    v = v * 10
    if k<5:
        fold_size = len(datasets) // 5
    else:
        fold_size = len(datasets) // k

    test_start = i * fold_size

    if i != k - 1 and i != 0:
        test_end = (i + 1) * fold_size
        TestSet = datasets[test_start:test_end]
        TrainSet = pd.concat([datasets[0:test_start], datasets[test_end:]])

    elif i == 0:
        test_end = fold_size
        TestSet = datasets[test_start:test_end]
        TrainSet = datasets[test_end:]

    else:
        TestSet = datasets[test_start:]
        TrainSet = datasets[0:test_start]

    return TrainSet.reset_index(drop=True), TestSet.reset_index(drop=True)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    dataset = shuffle(dataset)
    return dataset


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
	for i, data in enumerate(dataloader):
		siRNA = data[0].to(device)
		mRNA = data[1].to(device)
		siRNA_FM = data[2].to(device)
		mRNA_FM = data[3].to(device)
		label = data[4].to(device)
		td = data[6].to(device)
		pred,_,_ = model(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
		loss = criterion(pred[:,1],label.float())
		label = data[5]
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
	fpr, tpr, _ = roc_curve(label,pred)
	precision, recall,_= precision_recall_curve(label,pred)
	prauc=average_precision_score(label, pred)
	mcc=matthews_corrcoef(label,pred_cls)
	epoch_loss = running_loss.get_average()
	running_loss.reset()
	return epoch_loss, acc, sen, spe, pre, rec, f1score, rocauc, prauc, mcc, label, pred, pd.concat((pd.Series(fpr),pd.Series(tpr)),axis = 1) , pd.concat((pd.Series(precision),pd.Series(recall)),axis = 1)

def train_single(Args):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(Args.cuda)
	random.seed(Args.seed)
	os.environ['PYTHONHASHSEED']=str(Args.seed)
	np.random.seed(Args.seed)
	params = dict(
		data_root=Args.path,
		save_dir=Args.output_dir,
		dataset=Args.datasets[1],
		batch_size=Args.batch_size
	)
	logger = TrainLogger(params)
	dataset = pd.read_csv(Args.path + Args.datasets[0] + '.csv', dtype=str) 
	dataset = shuffle_dataset(dataset, Args.seed)
	test_df = pd.read_csv(Args.path + Args.datasets[1] + '.csv', dtype=str)
	for i_fold in range(Args.kfold):
		logger.info('*' * 50 + 'No.' + str(i_fold) + '-fold' + '*' * 50)
		train_df, valid_df = get_kfold_data_2(i_fold, dataset, Args.kfold)
		params = {'batch_size': Args.batch_size,
				'shuffle': True,
				'num_workers': 0,
				'drop_last': False}
		train_ds = DataLoader(data_process_loader(train_df.index.values, train_df.label.values,train_df.y.values, train_df, Args.datasets[0],Args.path), **params)
		valid_ds = DataLoader(data_process_loader(valid_df.index.values, valid_df.label.values,valid_df.y.values, valid_df, Args.datasets[0],Args.path),**params)
		test_ds = DataLoader(data_process_loader(test_df.index.values, test_df.label.values,test_df.y.values, test_df, Args.datasets[1],Args.path), **params)
		OFmodel = Oligo(vocab_size = Args.vocab_size, embedding_dim = Args.embedding_dim, lstm_dim = Args.lstm_dim,  n_head = Args.n_head, n_layers = Args.n_layers, lm1 = Args.lm1, lm2 = Args.lm2).to(device)
		if Args.resume is not None:
			OFmodel.load_state_dict(torch.load(Args.resume,map_location=device))
		criterion = nn.MSELoss() 
		best_AUC = 0.0
		best_loss = 1e10
		best_epoch = 0
		tolerence_epoch = Args.early_stopping
		optimizer = optim.Adam(OFmodel.parameters(), lr=Args.learning_rate)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = Args.weight_decay)
		logger.info(f"Number of train: {train_df.shape[0]}")
		logger.info(f"Number of val: {valid_df.shape[0]}")
		logger.info(f"Number of test: {test_df.shape[0]}")
		print('-----------------Start training!-----------------')
		running_loss = AverageMeter()
		for epoch in range(Args.epoch):
			for i, data in enumerate(train_ds):
				siRNA = data[0].to(device)
				mRNA = data[1].to(device)
				siRNA_FM = data[2].to(device)
				mRNA_FM = data[3].to(device)
				label = data[4].to(device)
				td = data[6].to(device)
				output,_,_ = OFmodel(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
				loss = criterion(output[:,1],label.float()) # 
				optimizer.zero_grad()
				loss.backward()
				running_loss.update(loss, output.shape[0])
				optimizer.step()
			scheduler.step()
			epoch_loss = running_loss.get_average()
			running_loss.reset()
			val_loss, val_acc, val_sen, val_spe, val_pre, val_rec, val_f1, val_rocauc, val_prauc, val_mcc, val_label, val_pred, auc_curve, prc_curve = val(OFmodel, criterion, valid_ds)
			if  val_loss < best_loss and val_rocauc > best_AUC: #
				best_AUC = val_rocauc
				best_loss = val_loss
				best_epoch = epoch
				msg = "epoch-%d, loss-%.4f, val_acc-%.4f,val_f1-%.4f, val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_f1-%.4f,val_loss-%.4f ***" % (epoch, epoch_loss, val_acc,val_f1,val_pre, val_rec,val_rocauc,val_prauc,val_f1,val_loss)
				torch.save(OFmodel.state_dict(), os.path.join(logger.get_model_dir(), msg+'.pth'))
			else:
				msg = "epoch-%d, loss-%.4f, val_acc-%.4f,val_f1-%.4f, val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_f1-%.4f,val_loss-%.4f " % (epoch, epoch_loss, val_acc,val_f1,val_pre, val_rec,val_rocauc,val_prauc,val_f1,val_loss)
			logger.info(msg)
			if epoch - best_epoch > tolerence_epoch:
				break

