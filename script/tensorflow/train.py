import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import sklearn
import random
from sklearn.utils import Bunch as _bunch
from sklearn.utils import Bunch
from tensorflow.keras.models import load_model
import pickle as pkl
from sklearn.model_selection import (train_test_split, GridSearchCV)
from tensorflow.python.keras import backend as K
from tensorflow.keras.utils import to_categorical
from model import PositionalEncoding,MultiHeadAttention,new_model
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score,roc_curve,auc,precision_recall_curve,matthews_corrcoef
from metrics import  sensitivity, specificity
from sklearn.model_selection import KFold
from tensorflow.python.keras.callbacks import ModelCheckpoint,Callback
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
import matplotlib.pyplot as plt
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
# session = tf.InteractiveSession(config=config)
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# tf.keras.backend.set_session(sess)
from train_logger import TrainLogger
import warnings
warnings.filterwarnings("ignore")


def OneHot(RNA):
    nucleotides = 'AGCU' 
    char_to_int = dict((c, i) for i, c in enumerate(nucleotides))
    char_to_int['X'] = 'X'
    integer_encoded = [char_to_int[i] for i in RNA]
    onehot = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(nucleotides))]
        if value != 'X':
        	letter[value] = 1
        onehot.append(letter)
    return onehot

def Embedding(df,LEN):
	print('df.shape:',df.shape)
	position = [0] * 1 + [1] * 9 + [0] * (LEN - 10)
	a = np.zeros((df.shape[0],df.shape[1],LEN))
	a[df != 0] = 1
	b = np.zeros((df.shape[0],df.shape[1],LEN))
	for n in range(df.shape[0]):
		#b[n] = np.r_[a[n], [position]]
		b[n] = a[n]
	b = b.transpose((0,2,1))
	return b

def Embedding_2(df,LEN,encode):
	position = [0] * 1 + [1] * 3 + [0] * (LEN - 4)
	a = np.zeros((df.shape[0],8,LEN))
	a[df != 0] = 1
	b = np.zeros((df.shape[0],9,LEN))
	for n in range(df.shape[0]):
		b[n] = np.r_[np.r_[a[n], encode[n]],[position]]
		#b[n] = np.r_[np.r_[a[n], encode[n]]]
	b = b.transpose((0,2,1))
	return b

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

def reshape_df(train_df,test_df,size):
	x_train, x_val, y_train, y_val = train_test_split(
		train_df.images,
		train_df.target,
		stratify=pd.Series(train_df.target),
		test_size=0.2,
		shuffle=True,
		random_state=42)
	# x_train = train_df.images
	# y_train = train_df.target
	# x_val = test_df.images
	# y_val = test_df.target
	x_test = test_df.images
	y_test = test_df.target

	xtrain = x_train.reshape(x_train.shape[0], 1,size[1],size[2]).astype('float32')
	xtest = x_test.reshape(x_test.shape[0],1,size[1],size[2]).astype('float32')
	xval = x_val.reshape(x_val.shape[0], 1,size[1],size[2]).astype('float32')
	ytrain = to_categorical(y_train, 2)
	ytest = to_categorical(y_test, 2)
	yval = to_categorical(y_val, 2)
	return xtrain, xtest, xval, ytrain, ytest, yval

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
	for i, (input, target) in enumerate(dataloader):
		input_var = tf.Variable(input)
		target_var = tf.Variable(target)
		pred = model(input_var)
		loss = criterion(target_var,pred)
		pred_cls = tf.argmax(pred,axis=1)
		pred_prob = pred #pred_prob = tf.nn.softmax(pred,axis=0)
		pred_list.append(tf.squeeze(pred_prob).cpu().numpy())
		pred_cls_list.append(tf.squeeze(pred_cls).cpu().numpy())
		label_list.append(target_var.numpy())
		running_loss.update(loss, target_var.shape[0])
	pred = np.concatenate(pred_list, axis=0)[:,1]
	pred_cls = np.concatenate(pred_cls_list, axis=0)
	label = np.concatenate(label_list, axis=0)[:,1]
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
	LEN = Args.input_size[1]
	dataset_dict = {}
	for dataset in Args.datasets:
		path = '/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer_19/data/' + dataset +'_siRNA.csv'
		siRNAseq = pd.read_csv(path, sep=',')
		RNA_pair = np.zeros((len(siRNAseq), Args.input_size[2], LEN))
		for n in range(len(siRNAseq)):
		    siRNA_encode = np.asarray(OneHot(siRNAseq.loc[n, 'siRNA'])).T
		    #mRNA_encode = np.asarray(OneHot(siRNAseq.loc[n, 'mRNA'])).T # (4, 63)
		    #thermodynamics_encode = np.asarray(siRNAseq.iloc[n, 4:23]).T.reshape(1,19)
		    RNA_pair[n] = siRNA_encode
		    #RNA_pair[n] = np.concatenate((siRNA_encode, mRNA_encode[:,:21],mRNA_encode[:,21:42],mRNA_encode[:,42:],thermodynamics_encode))
		    #RNA_pair[n] = np.concatenate((siRNA_encode,thermodynamics_encode))
		dataset_dict[dataset + 'embedding'] = Bunch(target=siRNAseq['label'].values,images=Embedding(RNA_pair,LEN))
	os.environ["CUDA_VISIBLE_DEVICES"] = str(Args.cuda)
	random.seed(Args.seed)
	os.environ['PYTHONHASHSEED']=str(Args.seed)
	np.random.seed(Args.seed)
	tf.random.set_seed(Args.seed)# tf.random.set_random_seed(Args.seed)
	#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=100, verbose=0, mode='auto')
	#checkpoint = ModelCheckpoint(logger.get_model_dir() + '/loss.h5', monitor='val_accuracy', verbose=1,save_best_only=True,mode='max',period=1)
	#checkpoint2 = ModelCheckpoint(logger.get_model_dir() + '/AUC.h5', monitor='val_loss', verbose=1,save_best_only=True,mode='min',period=1)
	#callbacks = [early_stopping,checkpoint]
	#train_dir = "./data/" + Args.datasets[0] + ".pkl"
	#test_dir = "./data/" + Args.datasets[1] + ".pkl"
	#xtrain, xtest, xval, ytrain, ytest, yval = reshape_df(train_dir,test_dir,Args.input_size)
	xtrain, xtest, xval, ytrain, ytest, yval = reshape_df(dataset_dict[Args.datasets[0] + 'embedding'],dataset_dict[Args.datasets[1] + 'embedding'],Args.input_size)
	resampled_steps_per_epoch = np.ceil(ytrain.shape[0] / Args.batch_size)

	if Args.new_model == True:
		print('new model!')
		OFmodel = new_model(Args.input_size)
	else:
		print('load existed model:' + Args.old_model)
		OFmodel = load_model(Args.old_model,custom_objects={'PositionalEncoding': PositionalEncoding,'MultiHeadAttention':MultiHeadAttention})
	criterion = BinaryCrossentropy(from_logits=True)
	train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain)).cache()
	train_ds = train_ds.batch(Args.batch_size)
	val_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
	val_ds = val_ds.batch(Args.batch_size)
	test_ds = tf.data.Dataset.from_tensor_slices((xtest, ytest)).cache()
	test_ds = test_ds.batch(Args.batch_size)
	best_AUC = 0.0
	best_loss = 1e10
	best_epoch = 0
	tolerence_epoch = Args.early_stopping
	#best_model = new_model(Args.input_size)
	exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=Args.learning_rate, decay_steps=200, decay_rate=Args.weight_decay)
	optimizer = Adam(exponential_decay)
	params = dict(
        data_root="/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer_19/data",
        save_dir="result",
        dataset=Args.datasets[1],
        batch_size=Args.batch_size
    )
	logger = TrainLogger(params)
	logger.info(f"Number of train: {len(xtrain)}")
	logger.info(f"Number of val: {len(xval)}")
	logger.info(f"Number of test: {len(xtest)}")
	print('-----------------Start training!-----------------')
	running_loss = AverageMeter()
	for epoch in range(Args.epoch):
		for i, (input, target) in enumerate(train_ds):
			input_var = tf.Variable(input)
			target_var = tf.Variable(target)
			with tf.GradientTape(persistent=True) as tape:
				output = OFmodel(input_var)
				loss = criterion(target_var,output)
			grads = tape.gradient(loss, OFmodel.trainable_variables)
			optimizer.apply_gradients(grads_and_vars=zip(grads, OFmodel.trainable_variables))
			running_loss.update(loss, output.shape[0]) 
			del tape
		epoch_loss = running_loss.get_average()
		running_loss.reset()
		val_loss, val_acc, val_sen, val_spe, val_pre, val_rec, val_f1, val_rocauc, val_prauc, val_mcc, val_label, val_pred = val(OFmodel, criterion, val_ds)
		test_loss, test_acc, test_sen, test_spe, test_pre, test_rec, test_f1, test_rocauc, test_prauc, test_mcc, test_label, test_pred = val(OFmodel, criterion, test_ds)
		if  val_loss < best_loss and val_rocauc > best_AUC: #
			best_AUC = val_rocauc
			best_loss = val_loss
			best_epoch = epoch
			#best_model.set_weights(OFmodel.get_weights())
			
			#msg = "epoch-%d, loss-%.4f, val_acc-%.4f,val_f1-%.4f, val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_loss-%.4f ***" % (epoch, epoch_loss, val_acc,val_f1,val_pre, val_rec,val_rocauc,val_prauc,val_loss)
			msg = "epoch-%d, loss-%.4f, val_loss-%.4f, val_rocauc-%.4f, test_loss-%.4f, test_rocauc-%.4f ***" % (epoch, epoch_loss, val_loss,val_rocauc,test_loss,test_rocauc)
		else:
			#msg = "epoch-%d, loss-%.4f, val_acc-%.4f,val_f1-%.4f, val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_loss-%.4f " % (epoch, epoch_loss, val_acc,val_f1,val_pre, val_rec,val_rocauc,val_prauc,val_loss)
			msg = "epoch-%d, loss-%.4f, val_loss-%.4f, val_rocauc-%.4f, test_loss-%.4f, test_rocauc-%.4f " % (epoch, epoch_loss, val_loss,val_rocauc,test_loss,test_rocauc)
		logger.info(msg)
		if epoch - best_epoch > tolerence_epoch:
			break

	# best_model.save(logger.get_model_dir() + '/best_loss.h5',save_format='h5')
	# plotAUC(best_model,xtest,ytest, logger.get_model_dir() + '/AUC_' + Args.datasets[1] + '.pdf')
	# plotPRC(best_model,xtest, ytest,  logger.get_model_dir() + '/PRC_' + Args.datasets[1] + '.pdf')



		
#		path = './data/' + dataset +'_siRNA.csv'
#		siRNAseq = pd.read_csv(path, sep=',')
#		mRNA_list = []
#		siRNA_list = []
#		encode_list = []
#		for n in range(len(siRNAseq)):
#		    target_mRNA = siRNAseq.loc[n, 'mRNA']
#		    target_siRNA = siRNAseq.loc[n, 'siRNA']
#		    arr1 = OneHot(target_mRNA)
#		    arr2 = OneHot(target_siRNA)
#		    arr = np.zeros((LEN, 4))
#		    
#		    for m in range(len(arr1)):
#		        if arr1[m] == arr2[m]:
#		            arr[m] = arr1[m]
#		        else:
#		            arr[m] = np.add(arr1[m], arr2[m])
#		    arr = np.asarray(arr).flatten(order='C')
#		    mRNA_list.append(np.asarray(arr1).flatten('C'))
#		    siRNA_list.append(np.asarray(arr2).flatten('C'))
#		    encode_list.append(np.asarray(arr).flatten('C'))
#		print(mRNA_list[0])
#		print(siRNA_list[0])
#		print(encode_list[0])
#		
#		siRNAseq['enc_mRNA'] = pd.Series(mRNA_list)
#		siRNAseq['enc_siRNA'] = pd.Series(siRNA_list)
#		siRNAseq['encoded'] = pd.Series(encode_list)
#		
#		im = np.zeros((len(siRNAseq), 8, LEN))
#		for n in range(len(siRNAseq)):
#		    arr1 = OneHot(siRNAseq.loc[n, 'siRNA'])
#		    arr1 = np.asarray(arr1).T
#		    arr2 = OneHot(siRNAseq.loc[n, 'mRNA'])
#		    arr2 = np.asarray(arr2).T
#		    arr = np.concatenate((arr1, arr2))
#		    im[n] = arr
#
#		embedding = Bunch(target=siRNAseq['label'].values,images=im)
#		
#		#write_pkl(embedding,"encoded8x21cd33withoutTsai_Sum.pkl")
#		#plt.imshow(embedding.images[0], cmap='Greys')
#		#plt.savefig('embedding.pdf')
#		
#		
#		embedding = Bunch(target=siRNAseq['label'].values,images=Embedding(im,LEN))
#		
#		#write_pkl(embedding,"encodedposition9x21cd33withoutTsai_Sum.pkl")
#		
#		encode_list = np.zeros((im.shape[0],5,LEN))
#		for n in range(im.shape[0]):
#		    for m in range(im.shape[2]):
#		        arr1 = im[n,0:4,m].tolist()
#		        arr2 = im[n,4:8,m].tolist()
#		        arr = []
#		        if arr1 == arr2:
#		            arr = [0,0,0,0,0]
#		        else:
#		            arr = np.add(arr1,arr2).tolist()
#		            if (arr == [1,1,0,0]) or (arr == [0,0,1,1]):
#		                arr.append(1)
#		            else:
#		                arr.append(-1)
#		        encode_list[n,:,m] = arr
#
#		embedding = Bunch(target=siRNAseq['label'].values,images=Embedding_2(im,LEN,encode_list))
#		write_pkl(embedding,"./data/" + dataset + ".pkl")
