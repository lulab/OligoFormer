import pandas as pd


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

taka = pd.read_csv('data/takayuki.txt',sep='\t',index_col=0)
taka['label'] = [int(i) for i in taka.iloc[:,-2] > 70]
data = taka.iloc[:,[0,-1]]
data['mRNA'] = [antiRNA(i) for i in data.iloc[:,0]]
data.columns = ['siRNA','label','mRNA']
data[['siRNA','mRNA','label']].to_csv('data/Taka_siRNA.csv',index=False)

