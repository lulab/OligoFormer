import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
path = '/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/result/20231214_115252_new********/log/train/Train.log'
loss = pd.read_csv(path,header=None,skiprows=4,names=['info','train_loss','val_loss','val_auc','test_loss','test_auc'])
train_loss = [float(i.split('-')[1])for i in loss['train_loss']]
valid_loss = [float(i.split('-')[1])for i in loss['val_loss']]

plt.figure(figsize=(10, 20)) 
mpl.rcParams['font.sans-serif'] = ['Arial']  
# mpl.rcParams['font.weight'] = 'bold'  
mpl.rcParams['font.size'] = 10
fig, ax1 = plt.subplots()
ax1.plot(train_loss,color = 'b',linewidth=2,label='train loss')
ax1.set_xlabel("epoch",fontweight='bold')
ax1.set_ylabel("train loss",fontweight='bold')

ax2 = ax1.twinx()
ax2.plot(valid_loss,color = 'y',linewidth=2,label='valid loss')
ax2.set_ylabel("valid loss",fontweight='bold')
ax2.text(65,0.0779,'*',fontsize=20)

fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
plt.title('Training curve of OligoFormer',fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/figure/OligoFormer_loss.png',dpi=600)
plt.close()



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# path = '/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/result/20231214_115252_new********/log/train/Train.log'
# loss = pd.read_csv(path,header=None,skiprows=4,names=['info','train_loss','val_loss','val_auc','test_loss','test_auc'])
# train_loss = [float(i.split('-')[1])for i in loss['train_loss']]
# valid_loss = [float(i.split('-')[1])for i in loss['val_loss']]


# # train_pp = pd.read_csv('../result/' + model_name + '/train_pp.txt',header = None)
# # valid_pp = pd.read_csv('../result/' + model_name + '/valid_pp.txt',header=None)

# plt.figure(figsize=(10, 10)) 
# mpl.rcParams['font.sans-serif'] = ['Arial']  
# mpl.rcParams['font.weight'] = 'bold'  
# mpl.rcParams['font.size'] = 10
# plt.plot(train_loss,color = 'b',linewidth=2,label='train loss')
# plt.plot(valid_loss,color = 'y',linewidth=2,label='valid loss')
# plt.legend(('train loss','valid loss'))
# plt.xlabel('epoch')
# plt.ylabel('loss')

# plt.title('OligoFormer loss change during training')
# plt.savefig('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/figure/OligoFormer_loss.png',dpi=600)
# plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.figure(figsize=(10, 8))
mpl.rcParams['font.sans-serif'] = ['Arial'] 
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.size'] = 20


data = pd.read_csv('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/nohup.out',sep='\t')
plt.plot(data['epoch'],data['lr']*10000,color='#8a94ae',zorder=0,linewidth = 6)
plt.scatter(data['epoch'],data['lr']*10000,color='#44587e',zorder=1,s = 50)
plt.xlabel('Epoch',fontweight='bold',fontsize = 21)
plt.ylabel('Learning rate / 10$^{-4}$',fontweight='bold',fontsize = 21)
plt.title('Learning rate change using Schedular',fontweight='bold',fontsize = 21)
plt.savefig('/mnt/inspurfs/user-fs/qhsky1/baiyilan/OligoFormer/lr.png',dpi=600,bbox_inches='tight')
plt.close()

def normalize(data):
    min_data = torch.min(data,dim=1)[0]
    for idx,j in enumerate(min_data):
        if j < 0:
            data[idx,:] += torch.abs(min_data[idx])
            min_data = torch.min(data,dim=1)[0]
    max_data = torch.max(data,dim=1)[0]
    maxmin = max_data - min_data
    if min_data.shape[0] == data.shape[0]:
        min_data = min_data.unsqueeze(1)
        maxmin = maxmin.unsqueeze(1)
    else:
        min_data = min_data.unsqueeze(0)
        maxmin = maxmin.unsqueeze(0)
    data = torch.sub(data,min_data).true_divide(maxmin)
    data = (data-0.5).true_divide(0.5)
    return data