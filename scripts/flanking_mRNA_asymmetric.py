import pandas as pd
import numpy as np
import re 
import copy
path = './'
_FLANK5 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,50,60,70,80,90,100]
_FLANK3 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,50,60,70,80,90,100]
# _FLANK5 = [1]
# _FLANK3 = [2]
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


Hu_mRNA = pd.read_csv(path + 'data/Hu_mRNA.fa',header=None)[1::2].reset_index(drop=True)
for i in range(Hu_mRNA.shape[0]):
    Hu_mRNA.iloc[i,0] = 'X' * 100 + Hu_mRNA.iloc[i,0].replace('T','U') + 'X' * 100 

new_mRNA = pd.read_csv(path + 'data/new_mRNA.fa',header=None)[1::2].reset_index(drop=True)
for i in range(new_mRNA.shape[0]):
    new_mRNA.iloc[i,0] = 'X' * 100 + new_mRNA.iloc[i,0].replace('T','U') + 'X' * 100 

Taka_mRNA = pd.read_csv(path + 'data/taka_mRNA.fa',header=None)[1::2].reset_index(drop=True)
for i in range(Taka_mRNA.shape[0]):
    Taka_mRNA.iloc[i,0] = 'X' * 100 + Taka_mRNA.iloc[i,0].replace('T','U') + 'X' * 100 

Hu_o = pd.read_csv(path + 'data/Hu.csv')
new_o = pd.read_csv(path + 'data/new.csv')
Taka_o = pd.read_csv(path + 'data/Taka.csv')

Hu = Hu_o.copy(deep=True)
new = new_o.copy(deep=True)
Taka = Taka_o.copy(deep=True)
Hu['flanking'] = Hu['y']
new['flanking'] = new['y']
Taka['flanking'] = Taka['y']


for FLANK5 in _FLANK5:
    for FLANK3 in _FLANK3:
        if FLANK5 == FLANK3:
            continue
        print('FLANK:',FLANK5,FLANK3)
        for i in range(Hu.shape[0]):
            count = 0
            for j in range(Hu_mRNA.shape[0]):
                res = re.search(antiRNA(Hu.iloc[i,0]),Hu_mRNA.iloc[j,0])
                if res is not None:
                    Hu.iloc[i,-1] = Hu_mRNA.iloc[j,0][(res.span(0)[0] - FLANK5) : (res.span(0)[1] + FLANK3)]
                    count += 1
            if count == 0:
                print(i,j,'here') 
        for i in range(new.shape[0]):
            count = 0
            for j in range(new_mRNA.shape[0]):
                res = re.search(antiRNA(new.iloc[i,0]),new_mRNA.iloc[j,0])
                if res is not None:
                    new.iloc[i,-1] = new_mRNA.iloc[j,0][(res.span(0)[0] - FLANK5) : (res.span(0)[1] + FLANK3)]
                    count += 1
                    break
            if count == 0:
                print(i,j,'here')
        for i in range(Taka.shape[0]):
            count = 0
            for j in range(Taka_mRNA.shape[0]):
                res = re.search(antiRNA(Taka.iloc[i,0]),Taka_mRNA.iloc[j,0])
                if res is not None:
                    Taka.iloc[i,-1] = Taka_mRNA.iloc[j,0][(res.span(0)[0] - FLANK5) : (res.span(0)[1] + FLANK3)]
                    count += 1
            if count > 1:
                print(i,j)
        # for i in `ls ./`; do mv $i/*.fa $i/fasta/ ; done
        with open(path + 'flanking_asymmetric/' + str(FLANK5) + '_' + str(FLANK3) + '/fasta/' + 'Hu_mRNA.fa','a') as f:
            for i in range(Hu.shape[0]):
                f.write('>RNA' + str(i) + '\n')
                f.write(Hu.iloc[i,-1] + '\n')
        Hu_o['mRNA'] = Hu['flanking']
        Hu_o.to_csv(path + 'flanking_asymmetric/' + str(FLANK5) + '_' + str(FLANK3) + '/Hu.csv',index=False)
        with open(path + 'flanking_asymmetric/' + str(FLANK5) + '_' + str(FLANK3) + '/fasta/' + 'new_mRNA.fa','a') as f:
            for i in range(new.shape[0]):
                f.write('>RNA' + str(i) + '\n')
                f.write(new.iloc[i,-1] + '\n')
        new_o['mRNA'] = new['flanking']
        new_o.to_csv(path + 'flanking_asymmetric/' + str(FLANK5) + '_' + str(FLANK3) + '/new.csv',index=False)
        with open(path + 'flanking_asymmetric/' + str(FLANK5) + '_' + str(FLANK3) + '/fasta/' + 'Taka_mRNA.fa','a') as f:
            for i in range(Taka.shape[0]):
                f.write('>RNA' + str(i) + '\n')
                f.write(Taka.iloc[i,-1] + '\n')
        Taka_o['mRNA'] = Taka['flanking']
        Taka_o.to_csv(path + 'flanking_asymmetric/' + str(FLANK5) + '_' + str(FLANK3) + '/Taka.csv',index=False)


