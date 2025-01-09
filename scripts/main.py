from train_single import train_single
from test_single import test_single
from test import test
from train import train
from infer import infer
from mismatch import mismatch
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
def main():
    parser = argparse.ArgumentParser(description='OligoFormer')
    # Data options
    parser.add_argument('--datasets', 		type=str, nargs='+', default=['Hu','Mix'], help="[Train,Test]: ['Hu','Mix','Taka']")
    parser.add_argument('--path',     		type=str, default='./data/', help='train or infer')
    parser.add_argument('--output_dir',  	type=str, default="result/", help='output directory')
    parser.add_argument('--best_model',  	type=str, default="./model/best_model.pth", help='best model directory')
    parser.add_argument('--resume',  	type=str, default=None, help='retrain on existing models')
    parser.add_argument('--cuda',     		type=int, default=0, help='ID number of GPU')
    parser.add_argument('--seed',     		type=int, default=42, help='random seed')


    # Training Hyper-parameters
    parser.add_argument('--learning_rate',  type=float, default=0.0001, help='learning rate') # 0.00001
    parser.add_argument('--batch_size',     type=int,   default=16, help='input batch size')
    parser.add_argument('--epoch',          type=int,   default=200, help='number of epochs to train')
    parser.add_argument('--early_stopping', type=int,   default=30, help='early stopping')
    parser.add_argument('--weight_decay',   type=float, default=0.999, help='weight decay')
    parser.add_argument('--kfold',    type=int, default=5, help='K-fold cross validation')

    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=26, help='vocab size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding hidden dimension')
    parser.add_argument('--lstm_dim', type=int, default=32, help='lstm dimension')
    parser.add_argument('--n_head', type=int, default=8, help='the number of heads')
    parser.add_argument('--n_layers', type=int, default=1, help='the number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--lm', type=int, default=19, help='flanking length')
    parser.add_argument('--lm1', type=int, default=19, help='left length')
    parser.add_argument('--lm2', type=int, default=19, help='right length')

    # option parameter
    parser.add_argument('-t','--test',action='store_true', help='test mode')
    parser.add_argument('-s','--single',action='store_true', help='single dataset mode')  

    # Infer module
    parser.add_argument('-i','--infer', type=int, default=0, help='0: None; 1: infer by fasta; 2: infer manually')
    parser.add_argument('-top','--top_n', type=int, default=-1, help='-1: all siRNAs; n: only calculate top n siRNAs')
    parser.add_argument('-i1','--infer_fasta', type=str, default='./data/example.fa', help='fasta file to infer')
    parser.add_argument('-i2','--infer_siRNA_fasta', nargs='?', const=False, help='siRNA fasta file to infer')
    
    # Functionality module
    parser.add_argument('-nf', '--no_func', action='store_true', help='not use filter based on known criteria determinant to functionality')
    
    # Off-target module
    parser.add_argument('-off','--off_target', action='store_true', help='evaluate off-target')
    parser.add_argument('-a','--all_human', action='store_true', help='scan all human UTR and ORF')
    parser.add_argument('--utr', type=str, default='./off-target/ref/human_UTR.txt', help='reference UTR sequence')
    parser.add_argument('--orf', type=str, default='./off-target/ref/human_ORF.txt', help='reference ORF sequence')
    parser.add_argument('--pita_threshold', type=float, default=-10, help='pita filter threshold')
    parser.add_argument('--targetscan_threshold', type=float, default=1, help='targetscan filter threshold')
    
    # Toxicity module
    parser.add_argument('-tox','--toxicity', action='store_true', help='evaluate toxicity')
    parser.add_argument('--toxicity_threshold', type=float, default=50.0, help='toxicity filter threshold')
    
    # Mismatch module
    parser.add_argument('-m','--mismatch', type=int, default=0, help='0: None; 1: compare two RNAs; 2: tranverse with 1,2 mismatches; 3: tranverse with 1,2,3 mismatches')

    Args = parser.parse_args()
    if not os.path.exists('./result'):
        os.mkdir('./result')

    if Args.mismatch > 0:
        mismatch(Args)
    elif Args.infer > 0:
        infer(Args)
    elif Args.test:
        test_single(Args) if Args.single else test(Args)
    else:
        train_single(Args) if Args.single else train(Args)

if __name__ == '__main__':
    main()

