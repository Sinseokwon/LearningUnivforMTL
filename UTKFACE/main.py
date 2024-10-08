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
parser.add_argument('--num_dummy', default=1, type=int, help='num of dummy')
parser.add_argument('--dummyweight', default=1e-6, type=float, help='weight for dummy')
parser.add_argument('--eps', default=1e-4, type=float, help='initialisation for auto-lambda')
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa, uncert, autol')
parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
opt = parser.parse_args()
torch.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)
def grad2vec(m, grads, grad_dims, task):
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.shared_modules():
        for p in mm.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

def compute_hessian(model, train_data, train_target,n,loss):
    age_loss = nn.CrossEntropyLoss()
    gen_loss = nn.CrossEntropyLoss()
    eth_loss = nn.CrossEntropyLoss()
    age, gen, eth = train_target[0].to(device), train_target[1].to(device), train_target[2].to(device)
    d = torch.autograd.grad(loss,model.n_dummy_parameter(n),retain_graph=True)
    grad = torch.autograd.grad(loss,model.encoder_parameter(),retain_graph=True)
    norm = torch.cat([w.view(-1) for w in d]).norm(2)
    eps = opt.eps / norm

    with torch.no_grad():
        for p,g in zip(model.n_dummy_parameter(n), d):
            p += eps *g
    _,train_pred_dummy_list = model(train_data)
    train_pred_dummy = train_pred_dummy_list[n]
    train_loss_dummy = [age_loss(train_pred_dummy[0],age), gen_loss(train_pred_dummy[1],gen),eth_loss(train_pred_dummy[2],eth)]
    dummyloss= sum(train_loss_dummy)
    grad_p = torch.autograd.grad(dummyloss,model.encoder_parameter())
    

    with torch.no_grad():
        for p,g in zip(model.n_dummy_parameter(n), d):
            p -= eps *g
    grad_p_scaled = [g * loss for g in grad_p] 
    grad_scaled = [g * dummyloss for g in grad]
    grad_p_sq = [g * dummyloss for g in grad_p]
    grad_sq = [g *loss for g in grad] 
    prox = [2*((sq+p_sq) - (p_scaled+scaled)) /(opt.eps)**2 for sq,p_sq,p_scaled,scaled in zip(grad_p_sq,grad_sq,grad_p_scaled,grad_scaled)]
    return prox


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
BATCH_SIZE = 64
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


if opt.weight=='autol':
    def train(train_loader,val_loader, model, optimizer, num_epoch):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        age_loss = nn.CrossEntropyLoss()
        gen_loss = nn.CrossEntropyLoss()
        eth_loss = nn.CrossEntropyLoss()
        train_batch = len(train_loader)

        for epoch in range(num_epoch):
            train_dataset = iter(train_loader)
            val_dataset = iter(val_loader)
            gen_correct, eth_correct, age_correct,total = 0,0,0,0   

            for k in range(train_batch):
                train_data, train_target = next(train_dataset)
                train_data = train_data.to(device)
                age, gen, eth = train_target[:,0].to(device), train_target[:,1].to(device), train_target[:,2].to(device)
                val_data, val_target = next(val_dataset)
                val_data = val_data.to(device)
                meta_optimizer.zero_grad()
                autol.unrolled_backward(train_data, train_target, val_data, val_target,
                                    scheduler.get_last_lr()[0], optimizer)
                meta_optimizer.zero_grad()
                optimizer.zero_grad()
                pred,pred_dummy = model(train_data)
 
                loss = age_loss(pred[0][0],age) + gen_loss(pred[0][1],gen) + eth_loss(pred[0][2],eth)
                train_loss = [age_loss(pred[0][0],age), gen_loss(pred[0][1],gen),eth_loss(pred[0][2],eth)]
                train_loss_tmp = [0] * 3
                train_loss_tmp = [w * train_loss[i] for i, w in enumerate(autol.meta_weights)]

                if opt.grad_method == 'none':
                    loss = sum(train_loss_tmp)
                    loss.backward(retain_graph=True)
                    for n in range(opt.num_dummy):
                        train_pred_dummy = train_pred_dummy_list[n]
                        train_loss_dummy = [compute_loss(train_pred_dummy[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
                        dummyloss = sum(train_loss_dummy)
                        prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                        for param,pr in zip(model.encoder_parameter(),prox):
                            param.grad += (opt.dummyweight/opt.num_dummy) * pr
                    optimizer.step()


                elif opt.grad_method == "pcgrad":
                    for i in range(3):
                        train_loss_tmp[i].backward(retain_graph=True)
                        grad2vec(model, grads, grad_dims, i)
                        model.zero_grad_shared_modules()
                    g = pcgrad(grads, rng, 3)
                    overwrite_grad(model, g, grad_dims, 3)
                    for n in range(opt.num_dummy):
                        train_pred_dummy = train_pred_dummy_list[n]
                        train_loss_dummy = [compute_loss(train_pred_dummy[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
                        dummyloss = sum(train_loss_dummy)
                        prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                        for param,pr in zip(model.encoder_parameter(),prox):
                            param.grad += (opt.dummyweight/opt.num_dummy) * pr
                    optimizer.step()

                elif opt.grad_method == "cagrad":
                    for i in range(3):
                        train_loss_tmp[i].backward(retain_graph=True)
                        grad2vec(model, grads, grad_dims, i)
                        model.zero_grad_shared_modules()
                    g = cagrad(grads, len(train_tasks), 0.4, rescale=1)
                    overwrite_grad(model, g, grad_dims, len(train_tasks))
                    for n in range(opt.num_dummy):
                        train_pred_dummy = train_pred_dummy_list[n]
                        train_loss_dummy = [compute_loss(train_pred_dummy[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
                        dummyloss = sum(train_loss_dummy)
                        prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                        for param,pr in zip(model.encoder_parameter(),prox):
                            param.grad += (opt.dummyweight/opt.num_dummy) * pr
                    optimizer.step()


                elif opt.grad_method == "nashmtl":
                    weight_method.backward(train_loss_tmp)
                    for n in range(opt.num_dummy):
                        train_pred_dummy = train_pred_dummy_list[n]
                        train_loss_dummy = [compute_loss(train_pred_dummy[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
                        dummyloss = sum(train_loss_dummy)
                        prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                        for param,pr in zip(model.encoder_parameter(),prox):
                            param.grad += (opt.dummyweight/opt.num_dummy) * pr
                    optimizer.step()


                elif opt.grad_method == "imtl":
                    weight_method.backward(train_loss_tmp)
                    for n in range(opt.num_dummy):
                        train_pred_dummy = train_pred_dummy_list[n]
                        train_loss_dummy = [compute_loss(train_pred_dummy[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
                        dummyloss = sum(train_loss_dummy)
                        prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                        for param,pr in zip(model.encoder_parameter(),prox):
                            param.grad += (opt.dummyweight/opt.num_dummy) * pr
                    optimizer.step()

            gen_acc, eth_acc, age_acc = gen_correct/total, eth_correct/total, age_correct/total
            print(f'Epoch : {epoch+1}/{num_epoch},    Age Accuracy : {age_acc*100},    Gender Accuracy : {gen_acc*100},    Ethnicity Accuracy : {eth_acc*100}\n')
else:
    def train(train_loader, model, optimizer, num_epoch):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        age_loss = nn.CrossEntropyLoss()
        gen_loss = nn.CrossEntropyLoss()
        eth_loss = nn.CrossEntropyLoss()
        train_batch = len(train_loader)

        for epoch in range(num_epoch):
            train_dataset = iter(train_loader)
            gen_correct, eth_correct, age_correct,total = 0,0,0,0 

            for k in range(train_batch):
                total += BATCH_SIZE
                train_data, train_target = next(train_dataset)
                train_data = train_data.to(device)
                age, gen, eth = train_target[0].to(device), train_target[1].to(device), train_target[2].to(device)
                

                optimizer.zero_grad()

                pred,pred_dummy = model(train_data)
                age_correct += (pred[0].argmax(1) == age).type(torch.float).sum().item()
                gen_correct += (pred[1].argmax(1) == gen).type(torch.float).sum().item()
                eth_correct += (pred[2].argmax(1) == eth).type(torch.float).sum().item()

                train_loss = [age_loss(pred[0],age), gen_loss(pred[1],gen),eth_loss(pred[2],eth)]
                train_loss_tmp = [0] * 3
                if opt.weight in ['equal', 'dwa']:
                    train_loss_tmp = [w * train_loss[i] for i, w in enumerate(lambda_weight[epoch])]
                if opt.weight == 'uncert':
                    train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]
                if opt.grad_method == 'none':
                    loss = sum(train_loss_tmp)
                    loss.backward(retain_graph=True)
                    for n in range(opt.num_dummy):
                        train_pred_dummy = pred_dummy[n]
                        train_loss_dummy = [age_loss(pred_dummy[n][0],age), gen_loss(pred_dummy[n][1],gen),eth_loss(pred_dummy[n][2],eth)]
                        train_loss_dummy_tmp = [0] * 3
                        train_loss_dummy_tmp = [w * train_loss_dummy[i] for i, w in enumerate(lambda_weight[epoch])]
                        dummyloss = sum(train_loss_dummy_tmp)
                        prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                        for param,pr in zip(model.encoder_parameter(),prox):
                            param.grad += (opt.dummyweight/opt.num_dummy) * pr
                    optimizer.step()

                elif opt.grad_method == "pcgrad":
                    for i in range(3):
                        train_loss_tmp[i].backward(retain_graph=True)
                        grad2vec(model, grads, grad_dims, i)
                        model.zero_grad_shared_modules()
                    g = pcgrad(grads, rng, 3)
                    overwrite_grad(model, g, grad_dims, 3)
                    for n in range(opt.num_dummy):
                        train_pred_dummy = pred_dummy[n]
                        train_loss_dummy = [age_loss(pred_dummy[n][0],age), gen_loss(pred_dummy[n][1],gen),eth_loss(pred_dummy[n][2],eth)]
                        dummyloss = sum(train_loss_dummy)
                        prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                        for param,pr in zip(model.encoder_parameter(),prox):
                            param.grad += (opt.dummyweight/opt.num_dummy) * pr
                    optimizer.step()

                elif opt.grad_method == "cagrad":
                    for i in range(3):
                        train_loss_tmp[i].backward(retain_graph=True)
                        grad2vec(model, grads, grad_dims, i)
                        model.zero_grad_shared_modules()
                    g = cagrad(grads, 3, 0.4, rescale=1)
                    overwrite_grad(model, g, grad_dims, 3)
                    for n in range(opt.num_dummy):
                        train_pred_dummy = pred_dummy[n]
                        train_loss_dummy = [age_loss(pred_dummy[n][0],age), gen_loss(pred_dummy[n][1],gen),eth_loss(pred_dummy[n][2],eth)]
                        dummyloss = sum(train_loss_dummy)
                        prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                        for param,pr in zip(model.encoder_parameter(),prox):
                            param.grad += (opt.dummyweight/opt.num_dummy) * pr
                    optimizer.step()


                elif opt.grad_method == "nashmtl":
                    weight_method.backward(train_loss_tmp)
                    for n in range(opt.num_dummy):
                        train_pred_dummy = pred_dummy[n]
                        train_loss_dummy = [age_loss(pred_dummy[n][0],age), gen_loss(pred_dummy[n][1],gen),eth_loss(pred_dummy[n][2],eth)]
                        dummyloss = sum(train_loss_dummy)
                        prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                        for param,pr in zip(model.encoder_parameter(),prox):
                            param.grad += (opt.dummyweight/opt.num_dummy) * pr
                    optimizer.step()


                elif opt.grad_method == "imtl":
                    weight_method.backward(train_loss_tmp)
                    for n in range(opt.num_dummy):
                        train_pred_dummy = pred_dummy[n]
                        train_loss_dummy = [age_loss(pred_dummy[n][0],age), gen_loss(pred_dummy[n][1],gen),eth_loss(pred_dummy[n][2],eth)]
                        dummyloss = sum(train_loss_dummy)
                        prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                        for param,pr in zip(model.encoder_parameter(),prox):
                            param.grad += (opt.dummyweight/opt.num_dummy) * pr
                    optimizer.step()
            gen_acc, eth_acc, age_acc = gen_correct/total, eth_correct/total, age_correct/total
            print(f'Epoch : {epoch+1}/{num_epoch},    Age Accuracy : {age_acc*100},    Gender Accuracy : {gen_acc*100},    Ethnicity Accuracy : {eth_acc*100}\n')


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
            pred,pred_dummy = model(X)

            age_acc += (pred[0].argmax(1) == age).type(torch.float).sum().item()
            gen_acc += (pred[1].argmax(1) == gen).type(torch.float).sum().item()
            eth_acc += (pred[2].argmax(1) == eth).type(torch.float).sum().item()

    age_acc /= size
    gen_acc /= size
    eth_acc /= size

    print(f"Age Accuracy : {age_acc*100}%,     Gender Accuracy : {gen_acc*100},    Ethnicity Accuracy : {eth_acc*100}\n")
    with open('results/mtl_{}_{}_{}_{}_{}_test_perform_dummy_random{}.txt'.format(opt.weight, opt.grad_method, opt.seed, 
                                                                                                 opt.dummyweight, opt.eps,opt.num_dummy),'w',encoding='UTF-8') as f:
        print(f"Age Accuracy : {age_acc*100}%,     Gender Accuracy : {gen_acc*100},    Ethnicity Accuracy : {eth_acc*100}\n", file=f)

opt = parser.parse_args()
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
model = TridentNN(7, 2, 5,opt.num_dummy)

if opt.weight == 'uncert':
    logsigma = torch.tensor([-0.7] * 3, requires_grad=True, device=device)
    params = list(model.Nodummy_parameter()) + [logsigma]
    logsigma_ls = np.zeros([opt.e, 3], dtype=np.float32)
if opt.weight in ['dwa', 'equal']:
    T = 2.0 
    lambda_weight = np.ones([opt.e, 3])
    params = model.Nodummy_parameter()
if opt.weight == 'autol':
    params = model.parameters()
    autol = AutoLambda(model, device, train_tasks, pri_tasks, 0.1)
    meta_weight_ls = np.zeros([opt.e, len(train_tasks)], dtype=np.float32)
    meta_optimizer = optim.Adam([autol.meta_weights], lr=1e-4)
if opt.grad_method in ['nashmtl', 'imtl']:
    weight_method = AbsWeighting(opt.grad_method, model, task_num=3, device=device)
if opt.grad_method != 'none':
    rng = np.random.default_rng()
    grad_dims = []
    for mm in model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), 3).to(device)

optimizer = torch.optim.Adam(params, lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = int(opt.e/3), gamma = 0.5,last_epoch=-1)


train(train_loader, model, optimizer, opt.e)
print('Finished training, running the testing script...\n \n')
test(test_loader, model)
