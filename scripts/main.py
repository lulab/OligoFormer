from train import run
import argparse

def main():
    parser = argparse.ArgumentParser(description='OligoFormer')
    # Data options
    parser.add_argument('--datasets', 		type=str, default=['Hu','new'], help="[Train,Test]: ['Hu','Sha','Taka','Sum']") #'Twelve','Ontarget'
    parser.add_argument('--output_dir',  	type=str, default="./", help='output directory')
    parser.add_argument('--best_model',  	type=str, default="", help='output directory')
    parser.add_argument('--new_model',  	type=bool, default=True, help='whether creating a new model or not')
    parser.add_argument('--old_model',  	type=str, default='./result/7116/best_loss.h5', help='whether creating a new model or not')
    parser.add_argument('--cuda',     		type=int, default=0, help='number of GPU')
    parser.add_argument('--seed',     		type=int, default=42, help='random seed')
    parser.add_argument('--mode',     		type=str, default='train', help='train or infer')

    # Training Hyper-parameters
    parser.add_argument('--learning_rate',  type=float, default=0.0001, help='learning rate') # 0.00001
    parser.add_argument('--batch_size',     type=int,   default=16, help='input batch size')
    parser.add_argument('--epoch',          type=int,   default=100, help='number of epochs to train')
    parser.add_argument('--weight_decay',   type=float, default=0.999, help='weight decay') # 0.9999
    parser.add_argument('--early_stopping', type=int,   default=40, help='early stopping')
    parser.add_argument('--kfold',    type=int, default=5, help='K-fold cross validation')

    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=26, help='vocab size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding hidden dimension')
    parser.add_argument('--lstm_dim', type=int, default=32, help='lstm dimension')
    parser.add_argument('--n_head', type=int, default=8, help='the number of heads')
    parser.add_argument('--n_layers', type=int, default=1, help='the number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

    Args = parser.parse_args()
    run(Args)


if __name__ == '__main__':
    main()
