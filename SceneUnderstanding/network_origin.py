import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from typing import Iterator
import torchvision.models.resnet as resnet
from create_network import *

class dummy_network_split(nn.Module):
    def __init__(self,trained_model,tasks,num_dummy):
        super(dummy_network_split, self).__init__()
        ch = [256, 512, 1024, 2048]
        self.num_dummy = num_dummy
        self.tasks = tasks
        self.shared_conv = trained_model.shared_conv
        self.shared_layer1 = trained_model.shared_layer1
        self.shared_layer2 = trained_model.shared_layer2
        self.shared_layer3 = trained_model.shared_layer3
        self.shared_layer4 = trained_model.shared_layer4
        self.decoders = trained_model.decoders
        self.decodersdummy = nn.ModuleList()
        # for i in range(self.num_dummy):
        #     self.decodersdummy.append(nn.ModuleList([DeepLabHead(ch[-1], self.tasks[t]) for t in self.tasks]))
        for i in range(self.num_dummy):
            decoder_list = []
            for t in self.tasks:
                decoder_list.append(DeepLabHead(ch[-1], self.tasks[t]))
            self.decodersdummy.append(nn.ModuleList(decoder_list))
        print(len(self.decodersdummy))
        # self.decodersdummy = nn.ModuleList([DeepLabHead(ch[-1], self.tasks[t]) for t in self.tasks])
    def forward(self, input):
        _, _, im_h, im_w = input.shape
        x = self.shared_conv(input)
        x = self.shared_layer1(x)
        x = self.shared_layer2(x)
        x = self.shared_layer3(x)
        x = self.shared_layer4(x)
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](x), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        
        out_dummy_list=[]
        for n in range(self.num_dummy):
            out_dummy = [0 for _ in self.tasks]
            for i, t in enumerate(self.tasks):
                out_dummy[i] = F.interpolate(self.decodersdummy[n][i](x), size=[im_h, im_w], mode='bilinear', align_corners=True)
                if t == 'normal':
                    out_dummy[i] = out_dummy[i] / torch.norm(out_dummy[i], p=2, dim=1, keepdim=True)
            out_dummy_list.append(out_dummy)
        return out,out_dummy_list


    def shared_modules(self):
        return [self.shared_conv,
                self.shared_layer1,
                self.shared_layer2,
                self.shared_layer3,
                self.shared_layer4]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def prepare_rep(self,rep):
        self.rep = rep
        return rep
        
    def encoder_parameter(self):
        return (p for n, p in self.named_parameters() if "decoder" not in n)
    def decoder_parameter(self):
        return (p for n, p in self.named_parameters() if "decoder" in n)

    def Nodummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" not in n)
    def dummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" in n)
    # def n_dummy_parameter(self,n,i):
    #     return self.decodersdummy[n][i].parameters()
    def n_dummy_parameter(self,n):
        return self.decodersdummy[n].parameters()

    def initialize_decodersdummy(self):
        for module in self.decodersdummy.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class dummy_network_split_(nn.Module):
    def __init__(self,trained_model,tasks):
        super(dummy_network_split_, self).__init__()
        ch = [256, 512, 1024, 2048]
        self.tasks = tasks
        self.shared_conv = trained_model.shared_conv
        self.shared_layer1 = trained_model.shared_layer1
        self.shared_layer2 = trained_model.shared_layer2
        self.shared_layer3 = trained_model.shared_layer3
        self.shared_layer4 = trained_model.shared_layer4
        self.decoders = trained_model.decoders
        self.decodersdummy= nn.ModuleList([DeepLabHead(ch[-1], self.tasks[t]) for t in self.tasks])
        # self.decodersdummy = nn.ModuleList([DeepLabHead(ch[-1], self.tasks[t]) for t in self.tasks])
    def forward(self, input):
        _, _, im_h, im_w = input.shape
        x = self.shared_conv(input)
        x = self.shared_layer1(x)
        x = self.shared_layer2(x)
        x = self.shared_layer3(x)
        x = self.shared_layer4(x)
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](x), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        
        out_dummy = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out_dummy[i] = F.interpolate(self.decodersdummy[i](x), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out_dummy[i] = out_dummy[i] / torch.norm(out_dummy[i], p=2, dim=1, keepdim=True)
        return out,out_dummy


    def shared_modules(self):
        return [self.shared_conv,
                self.shared_layer1,
                self.shared_layer2,
                self.shared_layer3,
                self.shared_layer4]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def prepare_rep(self,rep):
        self.rep = rep
        return rep
        
    def encoder_parameter(self):
        return (p for n, p in self.named_parameters() if "decoder" not in n)
    def decoder_parameter(self):
        return (p for n, p in self.named_parameters() if "decoder" in n)

    def Nodummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" not in n)
    def dummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" in n)
    # def n_dummy_parameter(self,n,i):
    #     return self.decodersdummy[n][i].parameters()
    def n_dummy_parameter(self):
        return self.decodersdummy.parameters()
    def initialize_decodersdummy(self):
        for module in self.decodersdummy.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class dummy_network_mtan(nn.Module):
    def __init__(self,trained_model,tasks,num_dummy):
        super(dummy_network_mtan, self).__init__()
        self.num_dummy = num_dummy
        self.tasks = tasks
        ch = [256, 512, 1024, 2048]
        self.shared_conv = trained_model.shared_conv
        self.shared_layer1_b = trained_model.shared_layer1_b
        self.shared_layer1_t = trained_model.shared_layer1_t

        self.shared_layer2_b = trained_model.shared_layer2_b
        self.shared_layer2_t = trained_model.shared_layer2_t

        self.shared_layer3_b = trained_model.shared_layer3_b
        self.shared_layer3_t = trained_model.shared_layer3_t

        self.shared_layer4_b = trained_model.shared_layer4_b
        self.shared_layer4_t = trained_model.shared_layer4_t

        self.encoder_att_1 = trained_model.encoder_att_1
        self.encoder_att_2 = trained_model.encoder_att_2
        self.encoder_att_3 = trained_model.encoder_att_3
        self.encoder_att_4 = trained_model.encoder_att_4

        self.encoder_block_att_1 =  trained_model.encoder_block_att_1 
        self.encoder_block_att_2 =  trained_model.encoder_block_att_2 
        self.encoder_block_att_3 =  trained_model.encoder_block_att_3 

        self.down_sampling = trained_model.down_sampling
        
        self.decoders = trained_model.decoders
        self.decodersdummy = nn.ModuleList()
        for i in range(self.num_dummy):
            self.decodersdummy.append(nn.ModuleList([DeepLabHead(ch[-1], self.tasks[t]) for t in self.tasks]))



    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )

    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(nn.Conv2d(in_channel, 4 * out_channel, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(4 * out_channel))
        return resnet.Bottleneck(in_channel, out_channel, downsample=downsample)

    def forward(self, x):
        _, _, im_h, im_w = x.shape

        # Shared convolution
        x = self.shared_conv(x)

        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)

        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]

        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]

        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]

        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]

        # Task specific decoders
        out = [0 for _ in self.tasks]
        # out_dummy = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](a_4[i]), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)

        # for i, t in enumerate(self.tasks):
        #     out_dummy[i] = F.interpolate(self.decodersdummy[i](a_4[i]), size=[im_h, im_w], mode='bilinear', align_corners=True)
        #     if t == 'normal':
        #         out_dummy[i] = out_dummy[i] / torch.norm(out_dummy[i], p=2, dim=1, keepdim=True)
        # return out,out_dummy

        # out = [0 for _ in self.tasks]
        # for i, t in enumerate(self.tasks):
        #     out[i] = F.interpolate(self.decoders[i](x), size=[im_h, im_w], mode='bilinear', align_corners=True)
        #     if t == 'normal':
        #         out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        
        out_dummy_list=[]
        for n in range(self.num_dummy):
            out_dummy = [0 for _ in self.tasks]
            for i, t in enumerate(self.tasks):
                out_dummy[i] = F.interpolate(self.decodersdummy[n][i](a_4[i]), size=[im_h, im_w], mode='bilinear', align_corners=True)
                if t == 'normal':
                    out_dummy[i] = out_dummy[i] / torch.norm(out_dummy[i], p=2, dim=1, keepdim=True)
            out_dummy_list.append(out_dummy)
        return out,out_dummy_list



    def shared_modules(self):
        return [self.shared_conv,
                self.shared_layer1_b,
                self.shared_layer1_t,
                self.shared_layer2_b,
                self.shared_layer2_t,
                self.shared_layer3_b,
                self.shared_layer3_t,
                self.shared_layer4_b,
                self.shared_layer4_t,
                self.encoder_block_att_1,
                self.encoder_block_att_2,
                self.encoder_block_att_3]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()
    def prepare_rep(self,rep):
        self.rep = rep
        return rep

    def encoder_parameter(self):
        return (p for n, p in self.named_parameters() if "shared" in n)
    def decoder_parameter(self):
        return (p for n, p in self.named_parameters() if "shared" not in n)

    def Nodummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" not in n)
    def dummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" in n)
    def get_last_shared(self):
        return []
    def n_dummy_parameter(self,n):
        return self.decodersdummy[n].parameters()




class ORIG_network_split(nn.Module):
    def __init__(self,trained_model,tasks):
        super(ORIG_network_split, self).__init__()
        ch = [256, 512, 1024, 2048]
        self.tasks = tasks
        self.shared_conv = trained_model.shared_conv
        self.shared_layer1 = trained_model.shared_layer1
        self.shared_layer2 = trained_model.shared_layer2
        self.shared_layer3 = trained_model.shared_layer3
        self.shared_layer4 = trained_model.shared_layer4

        self.decoders = trained_model.decoders
    def forward(self, input):
        _, _, im_h, im_w = input.shape
        x = self.shared_conv(input)
        x = self.shared_layer1(x)
        x = self.shared_layer2(x)
        x = self.shared_layer3(x)
        x = self.shared_layer4(x)
        out = [0 for _ in self.tasks]

        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](x), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    def shared_modules(self):
        return [self.shared_conv,
                self.shared_layer1,
                self.shared_layer2,
                self.shared_layer3,
                self.shared_layer4]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()
    def prepare_rep(self,rep):
        self.rep = rep
        return rep
        
    def encoder_parameter(self):
        return (p for n, p in self.named_parameters() if "decoder" not in n)
    def decoder_parameter(self):
        return (p for n, p in self.named_parameters() if "decoder" in n)

    def Nodummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" not in n)
    def dummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" in n)



class ORIG_network_mtan(nn.Module):
    def __init__(self,trained_model,tasks):
        super(ORIG_network_mtan, self).__init__()
        self.tasks = tasks
        ch = [256, 512, 1024, 2048]
        self.shared_conv = trained_model.shared_conv
        self.shared_layer1_b = trained_model.shared_layer1_b
        self.shared_layer1_t = trained_model.shared_layer1_t

        self.shared_layer2_b = trained_model.shared_layer2_b
        self.shared_layer2_t = trained_model.shared_layer2_t

        self.shared_layer3_b = trained_model.shared_layer3_b
        self.shared_layer3_t = trained_model.shared_layer3_t

        self.shared_layer4_b = trained_model.shared_layer4_b
        self.shared_layer4_t = trained_model.shared_layer4_t

        self.encoder_att_1 = trained_model.encoder_att_1
        self.encoder_att_2 = trained_model.encoder_att_2
        self.encoder_att_3 = trained_model.encoder_att_3
        self.encoder_att_4 = trained_model.encoder_att_4

        self.encoder_block_att_1 =  trained_model.encoder_block_att_1 
        self.encoder_block_att_2 =  trained_model.encoder_block_att_2 
        self.encoder_block_att_3 =  trained_model.encoder_block_att_3 

        self.down_sampling = trained_model.down_sampling
        
        self.decoders = trained_model.decoders
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(nn.Conv2d(in_channel, 4 * out_channel, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(4 * out_channel))
        return resnet.Bottleneck(in_channel, out_channel, downsample=downsample)
    def forward(self, x):
        _, _, im_h, im_w = x.shape

        # Shared convolution
        x = self.shared_conv(x)

        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)

        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]

        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]

        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]

        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]

        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](a_4[i]), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    def encoder_parameter(self):
        return (p for n, p in self.named_parameters() if "shared" in n)
    def decoder_parameter(self):
        return (p for n, p in self.named_parameters() if "shared" not in n)

    def Nodummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" not in n)
    def dummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" in n)
    def get_last_shared(self):
        return []
    def n_dummy_parameter(self,n):
        return self.decodersdummy[n].parameters()

class dummy_MTLVGG16(nn.Module):
    def __init__(self, trained_model,num_tasks):
        super(dummy_MTLVGG16, self).__init__()
        filter = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.num_tasks = num_tasks

        # define VGG-16 block
        self.network_block =trained_model.network_block
    

        # define classifiers here
        self.classifier = trained_model.classifier
        self.dummyclassifier = nn.ModuleList()
        for i in range(num_tasks):
            self.dummyclassifier.append(nn.Sequential(nn.Linear(filter[-1], 5)))
    
    def forward(self, x, task_id, dummy=False):
        for layer in self.network_block:
            if isinstance(layer, ConditionalBatchNorm2d):
                x = layer(x, task_id)
            else:
                x = layer(x)

        rep = F.adaptive_avg_pool2d(x, 1)
        pred = self.classifier[task_id](rep.view(rep.shape[0], -1))
        pred_dummy = self.dummyclassifier[task_id](rep.view(rep.shape[0], -1))
        if dummy:
            return pred_dummy
        else:
            return pred
    def shared_modules(self):
        return [p for n, p in self.named_modules() if "classifier" not in n and "bn" not in n] 


    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def prepare_rep(self,rep):
        self.rep = rep
        return rep

    def encoder_parameter(self):
        return (p for n, p in self.named_parameters() if "classifier" not in n and "bn" not in n)
    def decoder_parameter(self):
        return (p for n, p in self.named_parameters() if "classifier" in n)

    def Nodummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" not in n)
    def dummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" in n)
    def get_last_shared(self):
        return self.network_block.parameters()


class ORIG_MTLVGG16(nn.Module):
    def __init__(self, trained_model,num_tasks):
        super(ORIG_MTLVGG16, self).__init__()
        filter = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.num_tasks = num_tasks

        # define VGG-16 block
        self.network_block =trained_model.network_block

        # define classifiers here
        self.classifier = trained_model.classifier
    
    def forward(self, x, task_id,return_representation=False):
        for layer in self.network_block:
            if isinstance(layer, ConditionalBatchNorm2d):
                x = layer(x, task_id)
            else:
                x = layer(x)
        rep = F.adaptive_avg_pool2d(x, 1)
        pred = self.classifier[task_id](rep.view(x.shape[0], -1))
        return pred 
    def shared_modules(self):
        return [self.network_block]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def prepare_rep(self,rep):
        self.rep = rep
        return rep

    def encoder_parameter(self):
        return (p for n, p in self.named_parameters() if "classifier" not in n and "bn" not in n)
    def decoder_parameter(self):
        return (p for n, p in self.named_parameters() if "classifier" in n)

    def Nodummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" not in n)
    def dummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" in n)
    def get_last_shared(self):
        return self.network_block.parameters()



class dummy_MTLVGG16_v2(nn.Module):
    def __init__(self, trained_model,num_tasks):
        super(dummy_MTLVGG16, self).__init__()
        filter = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.num_tasks = num_tasks

        # define VGG-16 block
        self.network_block =trained_model.network_block
    

        # define classifiers here
        self.classifier = trained_model.classifier
        self.dummyclassifier = nn.ModuleList()
        for i in range(num_tasks):
            self.dummyclassifier.append(nn.Sequential(nn.Linear(filter[-1], 5)))
    
    def forward(self, x, task_id, dummy=False):
        for layer in self.network_block:
            if isinstance(layer, ConditionalBatchNorm2d):
                x = layer(x, task_id)
            else:
                x = layer(x)

        rep = F.adaptive_avg_pool2d(x, 1)
        pred = self.classifier[task_id](rep.view(rep.shape[0], -1))
        pred_dummy = self.dummyclassifier[task_id](rep.view(rep.shape[0], -1))
        if dummy:
            return pred_dummy
        else:
            return pred
    def shared_modules(self):
        return [p for n, p in self.named_modules() if "classifier" not in n and "bn" not in n] 


    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def prepare_rep(self,rep):
        self.rep = rep
        return rep

    def encoder_parameter(self):
        return (p for n, p in self.named_parameters() if "classifier" not in n and "bn" not in n)
    def decoder_parameter(self):
        return (p for n, p in self.named_parameters() if "classifier" in n)

    def Nodummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" not in n)
    def dummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" in n)
    def get_last_shared(self):
        return self.network_block.parameters()


class ORIG_MTLVGG16_v2(nn.Module):
    def __init__(self, trained_model,num_tasks):
        super(ORIG_MTLVGG16, self).__init__()
        filter = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.num_tasks = num_tasks

        # define VGG-16 block
        self.network_block =trained_model.network_block

        # define classifiers here
        self.classifier = trained_model.classifier
    
    def forward(self, x, task_id,return_representation=False):
        for layer in self.network_block:
            if isinstance(layer, ConditionalBatchNorm2d):
                x = layer(x, task_id)
            else:
                x = layer(x)
        rep = F.adaptive_avg_pool2d(x, 1)
        pred = self.classifier[task_id](rep.view(x.shape[0], -1))
        return pred 
    def shared_modules(self):
        return [self.network_block]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def prepare_rep(self,rep):
        self.rep = rep
        return rep

    def encoder_parameter(self):
        return (p for n, p in self.named_parameters() if "classifier" not in n and "bn" not in n)
    def decoder_parameter(self):
        return (p for n, p in self.named_parameters() if "classifier" in n)

    def Nodummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" not in n)
    def dummy_parameter(self):
        return (p for n, p in self.named_parameters() if "dummy" in n)
    def get_last_shared(self):
        return self.network_block.parameters()
