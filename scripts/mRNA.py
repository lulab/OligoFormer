import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score,roc_curve,auc,precision_recall_curve,matthews_corrcoef
from sklearn import preprocessing as p
min_max_scaler = p.MinMaxScaler()

def antiRNA(RNA):
    antiRNA = []
    for i in RNA:
        if i == 'A' or i == 'a':
            antiRNA.append('T')
        elif i == 'U' or i == 'u' or i == 'T' or i == 't':
            antiRNA.append('A')
        elif i == 'C' or i == 'c':
            antiRNA.append('G')
        elif i == 'G' or i == 'g':
            antiRNA.append('C')
    return ''.join(antiRNA[::-1])




taka = pd.read_csv('./data/iscore-taka.csv')
label = [int(i > 70) for i in taka['Activity']]
pred = taka['i-Score']
#pred = taka['s-Biopredsi']
#pred = pred.fillna(0) 
rocauc = roc_auc_score(label, pred)
print(rocauc)

Hu = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/Hu_siRNA.csv')
Sha = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/Sha_siRNA.csv')
Hu['y'] = Hu['label']
Sha['y'] = Sha['label']

Hu_old = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/old/Hu_siRNA_old.csv')
Sha_old = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/old/ShaDataset.csv')
taka = pd.read_csv('./data/takayuki.txt',sep='\t',index_col=0)
mRNA = ''.join([i[0] for i in taka.iloc[:,0]])
mRNA += taka.iloc[-1,0][1:]
tmRNA = antiRNA(mRNA)[::-1]
tmRNA = 'X' * 19 + tmRNA + 'X' * 19
# mRNA.to_csv('data/Taka_mRNA.csv',index=False,header=False)
taka['mRNA'] = taka.iloc[:,0]
for i in range(taka.shape[0]):
    taka.iloc[i,-1] = tmRNA[i:i+58]
Taka['y'] = [int(i > 70) for i in taka['Activity']]

Hu['label'] = Hu_old['efficacy']
Hu['label'] = min_max_scaler.fit_transform(np.array(Hu['label']).reshape(-1,1))
Sha['label'] = Sha_old['Activ']
Sha['label'] = 1 - min_max_scaler.fit_transform(np.array(Sha['label']).reshape(-1,1))

Taka['label'] = min_max_scaler.fit_transform(np.array(Taka['label']).reshape(-1,1))
Sum['label'] = 1 - min_max_scaler.fit_transform(np.array(Sum['label']).reshape(-1,1))

Hu.to_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/Hu_siRNA.csv',index=False)
Sha.to_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/Sha_siRNA.csv',index=False)
Taka.to_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/Taka_siRNA.csv',index=False)
Sum.to_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/Sum_siRNA.csv',index=False)

# Sha_old[Sha_old.iloc[:,3].duplicated()]
# Sha_old[Sha_old.iloc[:,3] == 'GGCAGAAUAAGUCUUCUCC']
# Sha_old[Sha_old.iloc[:,3] == 'CUUGAGGCUGUUGUCAUAC']
# Sha_old[Sha_old.iloc[:,3] == 'GGAGGCAUUGCUGAUGAUC']
# set(Sha[Sha['label'] == 1].iloc[:,0]) - set(Sha_old[Sha_old['Activ'] <= 30].iloc[:,3])
# Sha[Sha['siRNA'].duplicated()]
# Sha[Sha['siRNA'] == 'GGCAGAAUAAGUCUUCUCC']
# Sha[Sha['siRNA'] == 'CUUGAGGCUGUUGUCAUAC']
# Sha[Sha['siRNA'] == 'GGAGGCAUUGCUGAUGAUC']
# {'CAAGGGAGAACUGAGAAGA', 'AUUCUGCAUAUGACUGAUU', 'GUGCCGAGAAGAGGCUAAU'}


import pandas as pd
import numpy as np
Hu = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/Hu_siRNA.csv')
Sha = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/Sha_siRNA.csv')
Taka = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/Taka_siRNA.csv')
Sum = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/Sum_siRNA.csv')
new = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/new_siRNA.csv')


for i in range(Sha.shape[0]):
    with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/Sha_siRNA.fa','a') as f:
        f.write('>RNA' + str(i) + '\n')
        f.write(Sha['siRNA'][i] + '\n')
    with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/Sha_mRNA.fa','a') as f:
        f.write('>RNA' + str(i) + '\n')
        f.write(Sha['mRNA'][i] + '\n')

for i in range(Hu.shape[0]):
    with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/Hu_siRNA.fa','a') as f:
        f.write('>RNA' + str(i) + '\n')
        f.write(Hu['siRNA'][i] + '\n')
    with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/Hu_mRNA.fa','a') as f:
        f.write('>RNA' + str(i) + '\n')
        f.write(Hu['mRNA'][i] + '\n')

for i in range(Taka.shape[0]):
    with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/Taka_siRNA.fa','a') as f:
        f.write('>RNA' + str(i) + '\n')
        f.write(Taka['siRNA'][i] + '\n')
    with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/Taka_mRNA.fa','a') as f:
        f.write('>RNA' + str(i) + '\n')
        f.write(Taka['mRNA'][i] + '\n')

for i in range(Sum.shape[0]):
    with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/Sum_siRNA.fa','a') as f:
        f.write('>RNA' + str(i) + '\n')
        f.write(Sum['siRNA'][i] + '\n')
    with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/Sum_mRNA.fa','a') as f:
        f.write('>RNA' + str(i) + '\n')
        f.write(Sum['mRNA'][i] + '\n')

for i in range(new.shape[0]):
    with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/new_siRNA.fa','a') as f:
        f.write('>RNA' + str(i) + '\n')
        f.write(new['siRNA'][i] + '\n')
    with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/new_mRNA.fa','a') as f:
        f.write('>RNA' + str(i) + '\n')
        f.write(new['mRNA'][i] + '\n')

import pandas as pd
import numpy as np
sum_list = ['AP','Harborth','Hsieh','Khvorova','Reynolds','Ui-Tei','Vickers']
sum_dict = {}

for i in sum_list:
    sum_dict[i] = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/sumdata/' + i + '.txt',sep='\t')
    sum_dict[i].columns = ['siRNA','mRNA','label']
    sum_dict[i]['y'] = [int(j <= 30) for j in sum_dict[i]['label']]
    sum_dict[i]['label'] = 1- sum_dict[i]['label'] / 100
    sum_dict[i]['siRNA'] = [j.replace('T','U') for j in sum_dict[i]['siRNA']]
    sum_dict[i]['mRNA'] = [j.replace('T','U') for j in sum_dict[i]['mRNA']]
    sum_dict[i].to_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/' + i + '_siRNA.csv',index=False)

for i in sum_list:
    sum_dict[i] = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/reg/' + i + '_siRNA.csv')
    for j in range(sum_dict[i].shape[0]):
        with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/' + i + '_siRNA.fa','a') as f:
            f.write('>RNA' + str(j) + '\n')
            f.write(sum_dict[i]['siRNA'][j] + '\n')
        with open('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/data/fasta/' + i + '_mRNA.fa','a') as f:
            f.write('>RNA' + str(j) + '\n')
            f.write(sum_dict[i]['mRNA'][j] + '\n')