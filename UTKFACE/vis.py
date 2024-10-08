'''
    Main Script for Gender, Age, and Ethnicity identification on the cleaned UTK Dataset.

    The dataset can be found here (https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv)

    I implemented a MultiLabelNN that performs a shared high-level feature extraction before creating a 
    low-level neural network for each classification desired (gender, age, ethnicity).
'''
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
# torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
# Custom imports
from MultNN import TridentNN
import random
from weight_method import AbsWeighting
from auto_lambda import AutoLambda
from utils import *
import numpy as np
from torch.utils.data import random_split
import glob
from torch.utils.data import Dataset
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--e', '--num_epochs', default=100, type=int, help='(int) number of epochs to train the model')
parser.add_argument('--lr', '--learning_rate', default=0.00001, type=float, help='(float) learning rate of optimizer')
parser.add_argument('--seed', default=0, type=int, help='random seed ID')
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa, uncert, autol')
parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
parser.add_argument('--num_dummy', default=0, type=int, help='num of dummy')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
opt = parser.parse_args()
torch.manual_seed(2)
random.seed(2)
np.random.seed(2)



class UTKFace(Dataset):
    def __init__(self, image_paths):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        self.transform = transforms.Compose([transforms.Resize((64, 64)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std),
        ])

        self.image_paths = image_paths
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []

        for path in image_paths:
            filename = path[8:].split("_")
            if len(filename)==4:
                self.images.append(path)
                self.ages.append(int(filename[0]))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))
    
    def __len__(self):
         return len(self.images)

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert('RGB')

        img = self.transform(img)

        age = self.ages[index]
        age_category = pd.cut([age], bins=age_bins, labels=age_labels)
        gender = self.genders[index]
        race = self.races[index]
        return img,[age_category[0],gender,race]
BATCH_SIZE = 1024

image_paths = sorted(glob.glob("UTKFace/*.jpg.chip.jpg"))
TRAIN_SPLIT = 0.8
test_split = 0.2
num_train = round(TRAIN_SPLIT*len(image_paths))
num_val = round(test_split*len(image_paths))
(train_dataset, test_dataset) = random_split(image_paths,[num_train, num_val],generator=torch.Generator().manual_seed(opt.seed))
train_loader = DataLoader(UTKFace(train_dataset), shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(UTKFace(test_dataset), shuffle=False, batch_size=BATCH_SIZE)
age_bins = [0,10,20,30,40,50,60,120]
age_labels = [0, 1, 2, 3, 4, 5, 6 ]


def test(testloader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    size = len(testloader.dataset)
    model.eval()

    age_acc, gen_acc, eth_acc = 0, 0, 0  

    with torch.no_grad():
        for X, y in testloader:
            age, gen, eth = y[0].to(device), y[1].to(device), y[2].to(device)
            X = X.to(device)


from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def visualization(testloader, model):
  rep_list=[]
  age_list=[]
  gen_list=[]
  eth_list=[]
  size = len(testloader.dataset)
    # put the moel in evaluation mode so we aren't storing anything in the graph
  model.eval()
  with torch.no_grad():
    for epoch in range(1):
      dataset = iter(testloader)
      temp_df = pd.DataFrame()
      for k in range(1):
        X, y = next(dataset)
        fir, sec, thr = y[0].to(device), y[1].to(device), y[2].to(device)
        X = X.to(device)
        rep = model(X)
        rep_list.append(np.array(rep.cpu()))
        age_list.append(np.array(fir.cpu())) 
        gen_list.append(np.array(sec.cpu())) 
        eth_list.append(np.array(thr.cpu())) 
        rep_transformed = TSNE(n_components=2, learning_rate='auto',
               init='random', perplexity=3).fit_transform(rep_list[0])
      temp_df['first'] = rep_transformed[:,0]
      temp_df['second'] = rep_transformed[:,1]
      
      temp_df['age'] = age_list[0]
      temp_df['gen'] = gen_list[0]
      temp_df['eth'] = eth_list[0]

  return temp_df




device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
model = torch.load('models/model_{}_{}_{}_{}.pt'.format(opt.seed,opt.weight,opt.num_dummy,opt.grad_method))
class pruned_model(nn.Module):
    def __init__(self,model):
        super(pruned_model, self).__init__()
        # Construct the high level neural network
        self.features = model.features
        # Construct the low level neural networks
    def forward(self, x):
        rep = self.features(x).squeeze()
        return rep
rep_model = pruned_model(model)
rep_model.to(device)
df = visualization(test_loader, rep_model)
import matplotlib.pyplot as plt

colors = df['gen'].map({0:'red', 1: 'blue'})
plt.figure(figsize=(10, 8))
# 데이터프레임 시각화
for idx, row in df.iterrows():
    plt.scatter(row['first'], row['second'], c=colors[idx], edgecolor='none', s=50, label=None)
plt.savefig('visout/scatter_plot_{}_{}_{}_{}_sec_seedchange_2.png'.format(opt.seed,opt.weight,opt.num_dummy,opt.grad_method))
cmap = plt.cm.get_cmap('viridis', 7)  # viridis 컬러 맵, 7개의 색상

# 'fir' 열의 값에 맞게 색상을 지정
colors = df['age'].map({0: cmap(0), 1: cmap(1), 2: cmap(2), 3: cmap(3), 4: cmap(4), 5: cmap(5), 6: cmap(6)})

plt.figure(figsize=(10, 8))
# 데이터프레임 시각화
for idx, row in df.iterrows():
    plt.scatter(row['first'], row['second'], c=colors[idx], edgecolor='none', s=50, label=None)

plt.savefig('visout/scatter_plot_{}_{}_{}_{}_fir_seedchange_2.png'.format(opt.seed,opt.weight,opt.num_dummy,opt.grad_method))



cmap = plt.cm.get_cmap('tab10', 5)  # viridis 컬러 맵, 7개의 색상

# 'fir' 열의 값에 맞게 색상을 지정
colors = df['eth'].map({0: cmap(0), 1: cmap(1), 2: cmap(2), 3: 'purple', 4: "orange"})
plt.figure(figsize=(10, 8))
# 데이터프레임 시각화
for idx, row in df.iterrows():
    plt.scatter(row['first'], row['second'], c=colors[idx], edgecolor='none', s=50, label=None)

plt.savefig('visout/scatter_plot_{}_{}_{}_{}_thr_seedchange_2.png'.format(opt.seed,opt.weight,opt.num_dummy,opt.grad_method))