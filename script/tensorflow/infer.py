import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
#from model import PositionalEncoding,MultiHeadAttention

from keras_multi_head import MultiHeadAttention

import warnings
warnings.filterwarnings("ignore")

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len=None, embed_dim=None,**kwargs):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def call(self, a):
        position_embedding = np.array([
[pos / np.power(10000, 2. * i / self.embed_dim) for i in range(self.embed_dim)]
 for pos in range(self.seq_len)])
        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)
        return position_embedding + a
    def get_config(self):
        config = super().get_config().copy()
        config.update({
'seq_len' : self.seq_len,
'embed_dim' : self.embed_dim,
})
        return config


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
    return ''.join(antiRNA[::-1])
    
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



def infer(Args):
	LEN = Args.input_size[1]
	best_model = load_model('./result/0823/best_loss.h5',custom_objects={'PositionalEncoding': PositionalEncoding,'MultiHeadAttention':MultiHeadAttention})
	print('-----------------Start inferring!-----------------')
	if Args.mode == 'infer1':
		_mRNAs = pd.read_csv('./data/34mRNA.csv', sep=',')
		for _index in range(_mRNAs.shape[0]):
			_name = _mRNAs.iloc[_index,0].split('Rn_')[-1]
			print(_name)
			_mRNA = _mRNAs.iloc[_index,1]
			if len(_mRNA) < LEN:
			    raise Exception("The length of mRNA is less tha 19 nt!")
			_siRNA = list()
			for i in range(len(_mRNA) - LEN + 1): 
			    _siRNA.append(antiRNA(_mRNA[i:i+LEN]))
			_siRNA = pd.DataFrame(_siRNA)
			_siRNA.columns = ['siRNA']
			_RNA = np.zeros((len(_siRNA), Args.input_size[2], LEN))
			for n in range(len(_siRNA)):
			    siRNA_encode = np.asarray(OneHot(_siRNA.loc[n, 'siRNA'])).T
			    _RNA[n] = siRNA_encode
			xinfer = Embedding(_RNA,LEN)
			xinfer = xinfer.reshape(xinfer.shape[0],1,Args.input_size[1],Args.input_size[2]).astype('float32')
			Y_PRED = pd.DataFrame(best_model.predict(xinfer)[:,1])
			RESULT = pd.DataFrame()
			RESULT['pos'] = list(range(_siRNA.shape[0]))
			RESULT['sense'] = [antiRNA(_siRNA.iloc[i,0]) for i in range(_siRNA.shape[0])]
			RESULT['siRNA'] = _siRNA
			RESULT['efficacy'] = Y_PRED
			RESULT_ranked = RESULT.sort_values(by='efficacy', ascending=False)
			RESULT.to_csv('./result/' + Args.output_dir + '/34mRNA/' + str(_name) + '.txt',sep='\t',index = None,header=True)
			RESULT_ranked.to_csv('./result/' + Args.output_dir + '/34mRNA/' + str(_name) + '_ranked.txt',sep='\t',index = None,header=True)
			
	elif Args.mode == 'infer2':
		_mRNA = input("please input target mRNA: \n")
		if len(_mRNA) < LEN:
		    raise Exception("The length of mRNA is less tha 19 nt!")
		_siRNA = list()
		for i in range(len(_mRNA) - LEN + 1): 
		    _siRNA.append(antiRNA(_mRNA[i:i+LEN]))
		_siRNA = pd.DataFrame(_siRNA)
		_siRNA.columns = ['siRNA']
		_RNA = np.zeros((len(_siRNA), Args.input_size[2], LEN))
		for n in range(len(_siRNA)):
		    siRNA_encode = np.asarray(OneHot(_siRNA.loc[n, 'siRNA'])).T
		    _RNA[n] = siRNA_encode
		xinfer = Embedding(_RNA,LEN)
		xinfer = xinfer.reshape(xinfer.shape[0],1,Args.input_size[1],Args.input_size[2]).astype('float32')
		Y_PRED = pd.DataFrame(best_model.predict(xinfer)[:,1])
		RESULT = pd.DataFrame()
		RESULT['pos'] = list(range(_siRNA.shape[0]))
		RESULT['sense'] = [antiRNA(_siRNA.iloc[i,0]) for i in range(_siRNA.shape[0])]
		RESULT['siRNA'] = _siRNA
		RESULT['efficacy'] = Y_PRED
		RESULT_ranked = RESULT.sort_values(by='efficacy', ascending=False)
		RESULT.to_csv('./result/' + Args.output_dir + '/RESULT.txt',sep='\t',index = None,header=True)
		RESULT_ranked.to_csv('./result/' + Args.output_dir + '/RESULT_ranked.txt',sep='\t',index = None,header=True)
		



