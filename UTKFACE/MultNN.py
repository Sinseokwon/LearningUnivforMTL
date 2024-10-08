import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, resnet18, resnet34
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()   
    def forward(self, x):
        return x

class highLevelNN(nn.Module):
    def __init__(self):
        super(highLevelNN, self).__init__()
        self.CNN = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.CNN(x)

        return out

class lowLevelNN(nn.Module):
    def __init__(self, num_out):
        super(lowLevelNN, self).__init__()
        self.fc1 = nn.Linear(in_features=2048, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class TridentNN(nn.Module):
    def __init__(self, num_age, num_gen, num_eth,num_dummy):
        super(TridentNN, self).__init__()

        self.num_dummy = num_dummy
        self.CNN = resnet50(pretrained=True)
        self.CNN.fc = Identity()
        self.feature_layers = list(self.CNN.children())[:-1]
        self.features = nn.Sequential(*self.feature_layers)

        self.ageNN = lowLevelNN(num_out=num_age)
        self.genNN = lowLevelNN(num_out=num_gen)
        self.ethNN = lowLevelNN(num_out=num_eth)
        self.decodersdummy = nn.ModuleList()
        for i in range(self.num_dummy):
            self.decodersdummy.append(nn.ModuleList([lowLevelNN(num_out=num_age),lowLevelNN(num_out=num_gen),lowLevelNN(num_out=num_eth)]))
    def forward(self, x):
        rep = self.features(x).squeeze()
        age = self.ageNN(rep)
        gen = self.genNN(rep)
        eth = self.ethNN(rep)
        out_dummy_list=[]
        for n in range(self.num_dummy):
            dummy = []
            for i in range(3):
                dummy.append(self.decodersdummy[n][i](rep))
            out_dummy_list.append(dummy)
        return [age, gen, eth], out_dummy_list



    def shared_modules(self):
        return [self.CNN]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def prepare_rep(self):
        return self.rep
    
    def encoder_parameter(self):
        return (p for n, p in self.named_parameters() if "CNN" in n)
    def decoder_parameter(self):
        return (p for n, p in self.named_parameters() if "CNN" not in n)

    def Nodummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" not in n)
    def dummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" in n)
    # def n_dummy_parameter(self,n,i):
    #     return self.decodersdummy[n][i].parameters()
    def n_dummy_parameter(self,n):
        return self.decodersdummy[n].parameters()

if __name__ == '__main__':
    print('Testing out Multi-Label NN')
    mlNN = TridentNN(10, 10, 10)
    input = torch.randn(64, 1, 48, 48)
    y = mlNN(input)
    print(y[0].shape)
