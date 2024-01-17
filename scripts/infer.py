import os
import sys
import sklearn
import random
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import data_process_loader_infer
from torch.utils.data import DataLoader
from model import Oligo
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score,roc_curve,auc,precision_recall_curve,matthews_corrcoef
from metrics import  sensitivity, specificity
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import warnings
warnings.filterwarnings("ignore")

DeltaG = {'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33, 'CU': -2.08, 'AG': -2.08, 'CA': -2.11, 'UG': -2.11, 'GU': -2.24,  'AC': -2.24, 'GA': -2.35,  'UC': -2.35, 'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42, 'init': 4.09, 'endAU': 0.45, 'sym': 0.43}
DeltaH = {'AA': -6.82, 'UU': -6.82, 'AU': -9.38, 'UA': -7.69, 'CU': -10.48, 'AG': -10.48, 'CA': -10.44, 'UG': -10.44, 'GU': -11.40,  'AC': -11.40, 'GA': -12.44,  'UC': -12.44, 'CG': -10.64, 'GG': -13.39, 'CC': -13.39, 'GC': -14.88, 'init': 3.61, 'endAU': 3.72, 'sym': 0}

def antiRNA(RNA):
    antiRNA = []
    for i in RNA:
        if i == 'A' or i == 'a':
            antiRNA.append('U')
        elif i == 'U' or i == 'u' or i == 'T' or i == 't':
            antiRNA.append('A')
        elif i == 'C' or i == 'c':
            antiRNA.append('G')
        elif i == 'G' or i == 'g':
            antiRNA.append('C')
        elif i == 'X' or i == 'x':
            antiRNA.append('X')
    return ''.join(antiRNA[::-1])

def Calculate_DGH(seq):
    DG_all = 0
    DG_all += DeltaG['init']
    DG_all += ((seq[0] + seq[len(seq)-1]).count('A') + (seq[0] + seq[len(seq)-1]).count('U')) * DeltaG['endAU']
    DG_all += DeltaG['sym'] if antiRNA(seq).replace('T','U') == seq else 0
    for i in range(len(seq) - 1):
        DG_all += DeltaG[seq[i] + seq[i+1]]
    DH_all = 0
    DH_all += DeltaH['init']
    DH_all += ((seq[0] + seq[len(seq)-1]).count('A') + (seq[0] + seq[len(seq)-1]).count('U')) * DeltaH['endAU']
    DH_all += DeltaH['sym'] if antiRNA(seq).replace('T','U') == seq else 0
    for i in range(len(seq) - 1):
        DH_all += DeltaH[seq[i] + seq[i+1]]
    return DG_all,DH_all

def Calculate_end_diff(siRNA):
    count = 0
    _5 = siRNA[:2] # 5'end
    _3 = siRNA[-2:] # 3' end
    if _5 in ['AC','AG','UC','UG']:
        count += 1
    elif _5 in ['GA','GU','CA','CU']:
        count -= 1
    if _3 in ['AC','AG','UC','UG']:
        count += 1
    elif _3 in ['GA','GU','CA','CU']:
        count -= 1
    return float('{:.2f}'.format(DeltaG[_5] - DeltaG[_3] + count * 0.45))

def calculate_td(df):
	df['ends'] = df['siRNA']
	df['DG_1'] = df['siRNA']
	df['DH_1'] = df['siRNA']
	df['U_1'] = df['siRNA']
	df['G_1'] = df['siRNA']
	df['DH_all'] = df['siRNA']
	df['U_all'] = df['siRNA']
	df['UU_1'] = df['siRNA']
	df['G_all'] = df['siRNA']
	df['GG_1'] = df['siRNA']
	df['GC_1'] = df['siRNA']
	df['GG_all'] = df['siRNA']
	df['DG_2'] = df['siRNA']
	df['UA_all'] = df['siRNA']
	df['U_2'] = df['siRNA']
	df['C_1'] = df['siRNA']
	df['CC_all'] = df['siRNA']
	df['DG_18'] = df['siRNA']
	df['CC_1'] = df['siRNA']
	df['GC_all'] = df['siRNA']
	df['CG_1'] = df['siRNA']
	df['DG_13'] = df['siRNA']
	df['UU_all'] = df['siRNA']
	df['A_19'] = df['siRNA']

	for i in range(df.shape[0]):
		df['ends'] = [Calculate_end_diff(i) for i in df['siRNA']]
		df['DG_1'][i] = DeltaG[df.iloc[i,0][0:2]]
		df['DH_1'][i] = DeltaH[df.iloc[i,0][0:2]]
		df['U_1'][i] = int(df.iloc[i,0][0] == 'U')
		df['G_1'][i] = int(df.iloc[i,0][0] == 'G')
		df['DH_all'][i] = Calculate_DGH(df.iloc[i,0])[1]
		df['U_all'][i] = df.iloc[i,0].count('U') / 19
		df['UU_1'][i] = int(df.iloc[i,0][0:2] == 'UU')
		df['G_all'][i] = df.iloc[i,0].count('G') / 19
		df['GG_1'][i] = int(df.iloc[i,0][0:2] == 'GG')
		df['GC_1'][i] = int(df.iloc[i,0][0:2] == 'GC')
		df['GG_all'][i] = [df.iloc[i,0][j]+df.iloc[i,0][j+1] for j in range(18)].count('GG') / 18
		df['DG_2'][i] = DeltaG[df.iloc[i,0][1:3]]
		df['UA_all'][i] = [df.iloc[i,0][j]+df.iloc[i,0][j+1] for j in range(18)].count('UA') / 18
		df['U_2'][i] = int(df.iloc[i,0][1] == 'U')
		df['C_1'][i] = int(df.iloc[i,0][0] == 'C')
		df['CC_all'][i] = [df.iloc[i,0][j]+df.iloc[i,0][j+1] for j in range(18)].count('CC') / 18
		df['DG_18'][i] = DeltaG[df.iloc[i,0][17:19]]
		df['CC_1'][i] = int(df.iloc[i,0][0:2] == 'CC')
		df['GC_all'][i] = [df.iloc[i,0][j]+df.iloc[i,0][j+1] for j in range(18)].count('GC') / 18
		df['CG_1'][i] = int(df.iloc[i,0][0:2] == 'CG')
		df['DG_13'][i] = DeltaG[df.iloc[i,0][12:14]]
		df['UU_all'][i] = [df.iloc[i,0][j]+df.iloc[i,0][j+1] for j in range(18)].count('UU') / 18
		df['A_19'][i] = int(df.iloc[i,0][18] == 'A')
	df['td'] = [list(df.iloc[i,2:]) for i in range(df.shape[0])]
	return df[['siRNA','mRNA','td']]

def infer(Args):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(Args.cuda)
	random.seed(Args.seed)
	os.environ['PYTHONHASHSEED']=str(Args.seed)
	np.random.seed(Args.seed)
	best_model = Oligo(vocab_size = Args.vocab_size, embedding_dim = Args.embedding_dim, lstm_dim = Args.lstm_dim,  n_head = Args.n_head, n_layers = Args.n_layers).to(device)
	best_model.load_state_dict(torch.load("model/best_model.pth"))
	if not os.path.exists('./result'):
		os.mkdir('./result')
	print('-----------------Start inferring!-----------------')
	if Args.infer == 1:
		with open(Args.infer_fasta) as fa:
			fa_dict = {}
			for line in fa:
				line = line.replace('\n','')
				if line.startswith('>'):
					seq_name = line[1:]
					fa_dict[seq_name] = ''
				else:
					fa_dict[seq_name] += line.replace('\n','')
		for _name, _mRNA in fa_dict.items():
			print(_name)
			if len(_mRNA) < 19:
				raise Exception("The length of mRNA is less tha 19 nt!")
			_infer_df = pd.DataFrame(columns=['siRNA','mRNA'])
			_siRNA = list()
			for i in range(len(_mRNA) - 19 + 1): 
				_siRNA.append(antiRNA(_mRNA[i:i+19]))
			_infer_df['siRNA'] = _siRNA
			_cRNA = list()
			for i in range(len(_mRNA) - 19 + 1):
				_cRNA.append('X' * max(0, 19-i) + _mRNA[max(0,i-19):(i+38)] + 'X' * max(0,i+38-len(_mRNA)))
			_infer_df['mRNA'] = _cRNA
			_infer_df = calculate_td(_infer_df)
			if not os.path.exists('./data/infer'):
				os.mkdir('./data/infer')
			os.system('rm -rf ./data/infer/' + _name)
			os.system('mkdir ./data/infer/' + _name)
			for i in range(_infer_df.shape[0]):
				with open('./data/infer/' + _name + '/siRNA.fa','a') as f:
					f.write('>RNA' + str(i) + '\n')
					f.write(_infer_df['siRNA'][i] + '\n')
				with open('./data/infer/' + _name + '/mRNA.fa','a') as f:
					f.write('>RNA' + str(i) + '\n')
					f.write(_infer_df['mRNA'][i] + '\n')
			os.system('bash scripts/RNA-FM.sh ../../data/infer/' + _name)
			params = {'batch_size': 1,
				'shuffle': False,
				'num_workers': 0,
				'drop_last': False}
			infer_ds = DataLoader(data_process_loader_infer(_infer_df.index.values, _infer_df, _name),**params)
			Y_PRED = []
			for i, data in enumerate(infer_ds):
				siRNA = data[0].to(device)
				mRNA = data[1].to(device)
				siRNA_FM = data[2].to(device)
				mRNA_FM = data[3].to(device)
				td = data[4].to(device)
				pred,_,_ = best_model(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
				Y_PRED.append(pred[:,1].item())

			Y_PRED = [i*1.341 for i in Y_PRED]
			Y_PRED = pd.DataFrame(Y_PRED)
			RESULT = pd.DataFrame()
			RESULT['pos'] = list(range(_infer_df.shape[0]))
			RESULT['sense'] = [antiRNA(_infer_df.iloc[i,0]) for i in range(_infer_df.shape[0])]
			RESULT['siRNA'] = _infer_df['siRNA']
			print(Y_PRED)
			RESULT['efficacy'] = Y_PRED
			RESULT_ranked = RESULT.sort_values(by='efficacy', ascending=False)
			RESULT.to_csv(Args.infer_output + str(_name) + '.txt',sep='\t',index = None,header=True)
			RESULT_ranked.to_csv(Args.infer_output + str(_name) + '_ranked.txt',sep='\t',index = None,header=True)
		
	elif Args.infer == 2:
		_mRNA = input("please input target mRNA: \n")
		if len(_mRNA) < 19:
			raise Exception("The length of mRNA is less tha 19 nt!")
		_name = 'RNA0'
		print(_name)
		if len(_mRNA) < 19:
			raise Exception("The length of mRNA is less tha 19 nt!")
		_infer_df = pd.DataFrame(columns=['siRNA','mRNA'])
		_siRNA = list()
		for i in range(len(_mRNA) - 19 + 1): 
			_siRNA.append(antiRNA(_mRNA[i:i+19]))
		_infer_df['siRNA'] = _siRNA
		_cRNA = list()
		for i in range(len(_mRNA) - 19 + 1):
			_cRNA.append('X' * max(0, 19-i) + _mRNA[max(0,i-19):(i+38)] + 'X' * max(0,i+38-len(_mRNA)))
		_infer_df['mRNA'] = _cRNA
		_infer_df = calculate_td(_infer_df)
		if not os.path.exists('./data/infer'):
			os.mkdir('./data/infer')
		os.system('rm -rf ./data/infer/' + _name)
		os.system('mkdir ./data/infer/' + _name)
		for i in range(_infer_df.shape[0]):
			with open('./data/infer/' + _name + '/siRNA.fa','a') as f:
				f.write('>RNA' + str(i) + '\n')
				f.write(_infer_df['siRNA'][i] + '\n')
			with open('./data/infer/' + _name + '/mRNA.fa','a') as f:
				f.write('>RNA' + str(i) + '\n')
				f.write(_infer_df['mRNA'][i] + '\n')
		os.system('bash scripts/RNA-FM.sh ../../data/infer/' + _name)
		params = {'batch_size': 1,
			'shuffle': False,
			'num_workers': 0,
			'drop_last': False}
		infer_ds = DataLoader(data_process_loader_infer(_infer_df.index.values, _infer_df, _name),**params)
		Y_PRED = []
		for i, data in enumerate(infer_ds):
			siRNA = data[0].to(device)
			mRNA = data[1].to(device)
			siRNA_FM = data[2].to(device)
			mRNA_FM = data[3].to(device)
			td = data[4].to(device)
			pred,_,_ = best_model(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
			Y_PRED.append(pred[:,1].item())

		Y_PRED = [i*1.341 for i in Y_PRED]
		Y_PRED = pd.DataFrame(Y_PRED)
		RESULT = pd.DataFrame()
		RESULT['pos'] = list(range(_infer_df.shape[0]))
		RESULT['sense'] = [antiRNA(_infer_df.iloc[i,0]) for i in range(_infer_df.shape[0])]
		RESULT['siRNA'] = _infer_df['siRNA']
		print(Y_PRED)
		RESULT['efficacy'] = Y_PRED
		RESULT_ranked = RESULT.sort_values(by='efficacy', ascending=False)
		RESULT.to_csv(Args.infer_output + str(_name) + '.txt',sep='\t',index = None,header=True)
		RESULT_ranked.to_csv(Args.infer_output + str(_name) + '_ranked.txt',sep='\t',index = None,header=True)






