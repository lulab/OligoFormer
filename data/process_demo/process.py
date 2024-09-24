import os
import re
import itertools
import random
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')



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
		if i % 10 == 0:
			print(i)
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
	df['td'] = [list(df.iloc[i,4:]) for i in range(df.shape[0])]
	return df[['siRNA','mRNA','label','y','td']]


def main():
    parser = argparse.ArgumentParser(description='OligoFormer')
    # Data options
    parser.add_argument('--datasets', 		type=str, nargs='+', default=['Hu','Mix','Taka'], help="Hu,Mix,Taka")
    parser.add_argument('--input',  	type=str, default="./raw/", help='input directory')
    parser.add_argument('--output',  	type=str, default="./processed/", help='output directory')
    Args = parser.parse_args()
    for dataset in Args.datasets:
        data = pd.read_csv(Args.input + dataset + '.csv')
        data['y'] = [int(i > 0.7) for i in data['label']]
        data = calculate_td(data)
        data['td'] = [','.join([str(i) for i in data.iloc[j,-1]])  for j in range(data.shape[0])]
        data.to_csv(Args.output + dataset + '.csv',index=False)
        if not os.path.exists(Args.output + 'fasta'):
            os.mkdir(Args.output + 'fasta')
        with open(Args.output + 'fasta/' + dataset + '_mRNA.fa','a') as f:
            for i in range(data.shape[0]):
                f.write('>RNA' + str(i) + '\n')
                f.write(data.iloc[i,1] + '\n')
        f.close()
        with open(Args.output + 'fasta/' + dataset + '_siRNA.fa','a') as f:
            for i in range(data.shape[0]):
                f.write('>RNA' + str(i) + '\n')
                f.write(data.iloc[i,0] + '\n')
        f.close()

if __name__ == '__main__':
    main()


