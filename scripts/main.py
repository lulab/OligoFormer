from train import run
from infer import infer
import argparse

def main():
    parser = argparse.ArgumentParser(description='OligoFormer')
    # Data options    
    parser.add_argument('--datasets', 		type=str, default=['Hu','new'], nargs='+',help="[Train,Test]: ['Hu','new','Taka']") 
    parser.add_argument('--output_dir',  	type=str, default="./", help='output directory')
    parser.add_argument('--best_model',  	type=str, default="", help='output directory')
    parser.add_argument('--cuda',     		type=int, default=0, help='number of GPU')
    parser.add_argument('--seed',     		type=int, default=42, help='random seed')
    parser.add_argument('--mode',     		type=str, default='train', help='train or infer')
    parser.add_argument('--path',     		type=str, default='./data/', help='train or infer')

    # Training Hyper-parameters
    parser.add_argument('--learning_rate',  type=float, default=0.00001, help='learning rate') # 0.0001
    parser.add_argument('--batch_size',     type=int,   default=16, help='input batch size')
    parser.add_argument('--epoch',          type=int,   default=100, help='number of epochs to train')
    parser.add_argument('--weight_decay',   type=float, default=0.999, help='weight decay') # 0.9999
    parser.add_argument('--early_stopping', type=int,   default=30, help='early stopping')
    parser.add_argument('--kfold',    type=int, default=5, help='K-fold cross validation')

    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=26, help='vocab size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding hidden dimension')
    parser.add_argument('--lstm_dim', type=int, default=32, help='lstm dimension')
    parser.add_argument('--n_head', type=int, default=8, help='the number of heads')
    parser.add_argument('--n_layers', type=int, default=1, help='the number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

    # Infer mode
    parser.add_argument('--infer', type=int, default=0, help='0: train; 1: infer fasta; 2: infer manually')
    parser.add_argument('--infer_fasta', type=str, default='./data/example.fa', help='fasta file to infer')
    parser.add_argument('--offtarget', type=bool, default=False, help='whether to use off-target prediction')
    parser.add_argument('--infer_output', type=str, default='./result/', help='output path')
    Args = parser.parse_args()
    if Args.infer == 0 :
        run(Args)
    else:
        infer(Args)

if __name__ == '__main__':
    main()
