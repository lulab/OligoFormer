import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import itertools

def one_hot_encode(seq):
    mapping = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
    encoded_seq = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nucleotide in enumerate(seq):
        encoded_seq[i, mapping[nucleotide]] = 1
    return encoded_seq

class DNADataset(Dataset):
    def __init__(self, base_sequences, mismatch_sequences, labels):
        self.base_sequences = base_sequences
        self.mismatch_sequences = mismatch_sequences
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        base_seq = one_hot_encode(self.base_sequences[idx])
        mismatch_seq = one_hot_encode(self.mismatch_sequences[idx])
        label = self.labels[idx]
        return (base_seq, mismatch_seq), label

class MismatchModule(nn.Module):
    def __init__(self):
        super(MismatchModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 9 * 2, 64)  
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, base_seq, mismatch_seq):
        base_seq = self.conv1(base_seq)
        base_seq = torch.relu(base_seq)
        base_seq = self.pool(base_seq)
        mismatch_seq = self.conv1(mismatch_seq)
        mismatch_seq = torch.relu(mismatch_seq)
        mismatch_seq = self.pool(mismatch_seq)
        combined = torch.cat((base_seq.view(-1, 16 * 9), mismatch_seq.view(-1, 16 * 9)), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def calculate_kon(R1,R2):
    test_base_sequences = [R1] # base_seqs
    test_mismatch_sequences = [R2] # mismatch_seqs
    encoded_test_base_sequences = np.array([one_hot_encode(seq) for seq in test_base_sequences])
    encoded_test_mismatch_sequences = np.array([one_hot_encode(seq) for seq in test_mismatch_sequences])
    encoded_test_base_sequences = torch.tensor(encoded_test_base_sequences, dtype=torch.float32).transpose(1, 2)
    encoded_test_mismatch_sequences = torch.tensor(encoded_test_mismatch_sequences, dtype=torch.float32).transpose(1, 2)
    test_dataset = TensorDataset(encoded_test_base_sequences, encoded_test_mismatch_sequences)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for base_seqs, mismatch_seqs in test_dataloader:
        outputs = model(base_seqs, mismatch_seqs)
    return outputs.item()

def generate_mismatch_sequences(seq, mismatch_count):
    bases = ['A','U','C','G']
    mismatch_sequences = set()
    seq_len = len(seq)
    positions = list(itertools.combinations(range(seq_len), mismatch_count))
    for pos_combination in positions:
        for base_combination in itertools.product(bases, repeat=mismatch_count):
            if all(base_combination[i] != seq[pos_combination[i]] for i in range(mismatch_count)):
                new_seq = list(seq)
                for i in range(mismatch_count):
                    new_seq[pos_combination[i]] = base_combination[i]
                mismatch_sequences.add(''.join(new_seq))
    return list(mismatch_sequences)

model = MismatchModule()
model.load_state_dict(torch.load('mismatch_model.pth'))
model.eval()

def mismatch(Args):
    if Args.mismatch == 1:
        R1 = input("please input base RNA: \n")
        if len(R1) != 19:
            raise Exception("The length of base RNA is not 19 nt!")
        R2 = input("please input mismatched RNA: \n")
        if len(R2) != 19:
            raise Exception("The length of mismatched RNA is not 19 nt!")
        print(calculate_kon(R1,R2))
    else:
        bases = ['A','U','C','G']
        _Mismatch = []
        R1 = input("please input base RNA: \n")
        if len(R1) != 19:
            raise Exception("The length of base RNA is not 19 nt!")
        # for i in range(len(R1)):
        #     _Mismatch += [j + R1[:i] + R1[i+1:] for j in bases if j != R1[i]]
        _M1 = generate_mismatch_sequences(R1,1)
        _M1.sort()
        _Mismatch += _M1
        L1 = len(_Mismatch)
        _M2 = generate_mismatch_sequences(R1,2)
        _M2.sort()
        _Mismatch += _M2
        L2 = len(_Mismatch)
        if Args.mismatch == 3:
            _M3 = generate_mismatch_sequences(R1,3)
            _M3.sort()
            _Mismatch += _M3
            L3 = len(_Mismatch)
        data = pd.DataFrame((_Mismatch,_Mismatch,_Mismatch)).T
        data.columns = ['seq','kon','num_mismatch']
        count = 0
        for i in range(data.shape[0]):
            data.iloc[i,1] = calculate_kon(R1,data.iloc[i,0])
            if Args.mismatch == 2:
                if i % 156 == 0:
                    print(str(count) + '%')
                    count += 10
            else:
                if i % 2776 == 0:
                    print(str(count) + '%')
                    count += 10
            if i < L1:
                data.iloc[i,2] = 1
            elif i < L2:
                data.iloc[i,2] = 2
            else:
                data.iloc[i,2] = 3
        data.to_csv(Args.output_dir + '/result/mismatch.txt',sep='\t',index=False)


