import torch
import os
import sys
import sklearn
import random
import pickle as pkl
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import data_process_loader
from torch.utils.data import DataLoader
from model import Oligo,Oligo2,Oligo3
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score,roc_curve,auc,precision_recall_curve,matthews_corrcoef
from metrics import  sensitivity, specificity
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib as mpl
from train_logger import TrainLogger
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def find_metrics_best_for_shuffle(label, prob, cut_spe=0.95):
    fpr, tpr, _ = roc_curve(label, prob)
    a = 1 - fpr
    b = tpr
    Sensitivity = b
    Specificity = a
    Sensitivity_ = Sensitivity[Specificity >= cut_spe]
    if (len(Sensitivity_) == 1) & (Sensitivity_[0] == 0):
        Sensitivity_best = ((Sensitivity[1] - Sensitivity[0]) / (Specificity[1] - Specificity[0])) * cut_spe + Sensitivity[1] - ((Sensitivity[1] - Sensitivity[0]) / (Specificity[1] - Specificity[0])) *  Specificity[1]
    else:
        Sensitivity_best = np.max(Sensitivity_)
    return Sensitivity_best, Sensitivity, Specificity

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

def write_pkl(pkl_data,pkl_name):
	pkl_file = open(pkl_name, "wb")
	pkl.dump(pkl_data, pkl_file)
	pkl_file.close()

def plotPRC(model,X,Y,name):
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	y_pred = model.predict(X)[:,1]
	precision, recall, threshold = precision_recall_curve(Y[:,1],y_pred)
	prc = auc(recall, precision)
	plt.plot(recall, precision,label='OligoFormer (PRC: %s \u00B1 0.001)' % (np.round(prc, 3)))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='best')
	plt.savefig(name)
	Y_TRUE = pd.DataFrame(Y)
	Y_PRED = pd.DataFrame(model.predict(X)[:,1])
	with open(name.split('PRC')[0] + 'test_prediction.txt', 'w') as f:
		for i in range(Y_TRUE.shape[0]):
			f.write(str(Y_TRUE.iloc[i,1]) + " " + str(Y_PRED.iloc[i,0]) + '\n')

def plotAUC(model,X,Y,name):
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	y_pred = model.predict(X)[:,1]
	fpr, tpr, threshold = roc_curve(Y[:,1],y_pred)
	roc = auc(fpr, tpr)
	plt.plot(fpr, tpr,label='OligoFormer (AUC: %s \u00B1 0.001)' % (np.round(roc, 3)))
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.legend(loc='best')
	plt.savefig(name)

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
		# siRNA_ss = data[7].to(device)	
		# ss = data[8].to(device)
		pred,_,_ = model(siRNA,mRNA,siRNA_FM,mRNA_FM,td) #  siRNA_ss,ss
		loss = criterion(pred[:,1],label.float()) #criterion(pred,label)
		label = np.array([int(i > 0.7) for i in label])
		# label = data[5]
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
	return epoch_loss, acc, sen, spe, pre, rec, f1score, rocauc, prauc, mcc, label, pred

def run(Args):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(Args.cuda)
	random.seed(Args.seed)
	os.environ['PYTHONHASHSEED']=str(Args.seed)
	np.random.seed(Args.seed)
	train_df = pd.read_csv(Args.path + Args.datasets[0] + '.csv', dtype=str)
	# train_df,valid_df = train_test_split(train_df,test_size=0.2,shuffle=True,random_state=42)
	# train_df = train_df.reset_index(drop=True)
	# valid_df = valid_df.reset_index(drop=True)
	valid_df = pd.read_csv(Args.path + Args.datasets[1] + '.csv',  dtype=str)
	test_df = pd.read_csv(Args.path + Args.datasets[1] + '.csv', dtype=str)
	params = {'batch_size': Args.batch_size,
			'shuffle': True,
			'num_workers': 0,
			'drop_last': False}
	if not os.path.exists('./data/RNAFM'):
		os.system('bash scripts/RNA-FM-feature.sh')
	train_ds = DataLoader(data_process_loader(train_df.index.values, train_df.label.values,train_df.y.values, train_df, Args.datasets[0]), **params)
	valid_ds = DataLoader(data_process_loader(valid_df.index.values, valid_df.label.values,valid_df.y.values, valid_df, Args.datasets[1]),**params)
	test_ds = DataLoader(data_process_loader(test_df.index.values, test_df.label.values,test_df.y.values, test_df, Args.datasets[1]), **params)
	OFmodel = Oligo(vocab_size = Args.vocab_size, embedding_dim = Args.embedding_dim, lstm_dim = Args.lstm_dim,  n_head = Args.n_head, n_layers = Args.n_layers).to(device)
	# tmp = torch.load("model/best_model.pth")
	# OFmodel.load_state_dict(tmp)
	criterion = nn.MSELoss() #nn.BCELoss() # nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1)) #
	best_AUC = 0.0
	best_loss = 1e10
	best_epoch = 0
	tolerence_epoch = Args.early_stopping
	saliency_map = np.zeros((19,5))
	optimizer = optim.Adam(OFmodel.parameters(), lr=Args.learning_rate)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = Args.weight_decay)
	params = dict(
		data_root="/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data",
		save_dir="result",
		dataset=Args.datasets[1],
		batch_size=Args.batch_size
	)
	logger = TrainLogger(params)
	logger.info("pyTorch!!!")
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
			# siRNA_ss = data[7].to(device)	
			# ss = data[8].to(device)
			siRNA.requires_grad = True
			output, siRNA_attention, mRNA_attention = OFmodel(siRNA,mRNA,siRNA_FM,mRNA_FM,td) # siRNA_ss,ss
			np.save(logger.get_model_dir()+"/attention_"+str(epoch)+"_siRNA.npy",siRNA_attention.detach().cpu().numpy())
			np.save(logger.get_model_dir()+"/attention_"+str(epoch)+"_mRNA.npy",mRNA_attention.detach().cpu().numpy())
			loss = criterion(output[:,1],label.float()) # criterion(output, label)
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			gradients = siRNA.grad.detach().cpu().numpy().squeeze()
			
			saliency_map += gradients.sum(axis=0)
			running_loss.update(loss, output.shape[0])
			optimizer.step()
		# from scipy.ndimage import gaussian_filter
		# blurred_saliency_map = gaussian_filter(saliency_map.squeeze(), sigma=0.01)
		# np.save('./grad.npy',blurred_saliency_map)
		# mpl.rcParams['font.sans-serif'] = ['Arial']  
		# mpl.rcParams['font.weight'] = 'bold'  
		# mpl.rcParams['font.size'] = 12
		# fig = plt.figure()
		# ax = fig.add_subplot(1, 1, 1)
		# plt.imshow(blurred_saliency_map.T, cmap="RdYlBu", aspect="equal", interpolation="none")
		# ax.set_xticks(list(range(19)))
		# ax.set_yticks(list(range(5)))
		# ax.set_yticklabels(['A', 'U', 'C', 'G', 'X'], fontsize=12)
		# print(siRNA[0][0])
		# ax.set_title('Saliency Map for RNA Sequence', fontweight='bold')
		# plt.savefig('./mine4.png',dpi=600)
		# break
		scheduler.step()
		epoch_loss = running_loss.get_average()
		running_loss.reset()
		val_loss, val_acc, val_sen, val_spe, val_pre, val_rec, val_f1, val_rocauc, val_prauc, val_mcc, val_label, val_pred = val(OFmodel, criterion, valid_ds)
		test_loss, test_acc, test_sen, test_spe, test_pre, test_rec, test_f1, test_rocauc, test_prauc, test_mcc, test_label, test_pred = val(OFmodel, criterion, test_ds)
		if  val_loss < best_loss and val_rocauc > best_AUC: #
			best_AUC = val_rocauc
			best_loss = val_loss
			best_epoch = epoch
			#msg = "epoch-%d, loss-%.4f, val_acc-%.4f,val_f1-%.4f, val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_loss-%.4f ***" % (epoch, epoch_loss, val_acc,val_f1,val_pre, val_rec,val_rocauc,val_prauc,val_loss)
			msg = "epoch-%d, loss-%.4f, val_loss-%.4f, val_rocauc-%.4f,val_f1-%.4f, test_loss-%.4f, test_rocauc-%.4f, test_f1-%.4f ***" % (epoch, epoch_loss, val_loss, val_rocauc,val_f1, test_loss,test_rocauc,test_f1)
			torch.save(OFmodel.state_dict(), os.path.join(logger.get_model_dir(), msg+'.pth'))
		else:
			#msg = "epoch-%d, loss-%.4f, val_acc-%.4f,val_f1-%.4f, val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_loss-%.4f " % (epoch, epoch_loss, val_acc,val_f1,val_pre, val_rec,val_rocauc,val_prauc,val_loss)
			msg = "epoch-%d, loss-%.4f, val_loss-%.4f, val_rocauc-%.4f,val_f1-%.4f, test_loss-%.4f, test_rocauc-%.4f, test_f1-%.4f " % (epoch, epoch_loss, val_loss,val_rocauc,val_f1, test_loss,test_rocauc,test_f1)
			torch.save(OFmodel.state_dict(), os.path.join(logger.get_model_dir(), msg+'.pth'))
		logger.info(msg)
		if epoch - best_epoch > tolerence_epoch:
			break

