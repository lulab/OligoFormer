#from torch_train import run
from train import run
from infer import infer
import argparse

def main():
    parser = argparse.ArgumentParser(description='OligoFormer')
    # Data options
    parser.add_argument('--datasets', 		type=str, default=['Hu','Sha'], help="[Train,Test]: ['Hu','Sha','Taka','Sum']") #'Twelve','Ontarget'
    parser.add_argument('--output_dir',  	type=str, default="1204_taka", help='output directory')
    parser.add_argument('--best_model',  	type=str, default="", help='output directory')
    parser.add_argument('--cuda',     		type=int, default=0, help='number of GPU')
    parser.add_argument('--seed',     		type=int, default=42, help='random seed')
    parser.add_argument('--mode',     		type=str, default='train', help='train or infer')
    parser.add_argument('--best_save',		type=str, default='loss', help='loss or AUC')
    # Training Hyper-parameters
    # parser.add_argument('--lr_scheduler',   default="warmup", help=' lr scheduler: warmup/cosine')
    parser.add_argument('--new_model',  	type=bool, default=True, help='whether creating a new model or not')
    parser.add_argument('--old_model',  	type=str, default='./result/7116/best_loss.h5', help='whether creating a new model or not')
    parser.add_argument('--learning_rate',  type=float, default=0.0001, help='learning rate') # 0.00001
    parser.add_argument('--batch_size',     type=int,   default=16, help='input batch size')
    parser.add_argument('--epoch',          type=int,   default=200, help='number of epochs to train')
    parser.add_argument('--weight_decay',   type=float, default=0.999, help='weight decay') # 0.9999
    parser.add_argument('--early_stopping', type=int,   default=30, help='early stopping')

    # Model parameters
    parser.add_argument('--input_size', type=tuple, default=(1,19,4), help='shape')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

    Args = parser.parse_args()
    if (Args.mode == 'infer1') or (Args.mode == 'infer2'):
    	infer(Args)
    else:
    	run(Args)


if __name__ == '__main__':
    main()
