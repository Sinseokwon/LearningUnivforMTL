import argparse

import torch.optim as optim
import torch.utils.data.sampler as sampler

from auto_lambda import AutoLambda
from create_network import *
from create_dataset import *
from utils import *
from weight_method2 import AbsWeighting
from network_origin import *
parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--network', default='split', type=str, help='split, mtan')
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa, uncert, autol')
parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--with_noise', action='store_true', help='with noise prediction task')
parser.add_argument('--autol_init', default=0.1, type=float, help='initialisation for auto-lambda')
parser.add_argument('--autol_lr', default=1e-4, type=float, help='learning rate for auto-lambda')
parser.add_argument('--task', default='all', type=str, help='primary tasks, use all for MTL setting')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
parser.add_argument('--seed', default=0, type=int, help='random seed ID')
parser.add_argument('--epoch', default=300, type=int, help='num of epoch')
parser.add_argument('--num_dummy', default=1, type=int, help='num of dummy')
parser.add_argument('--dummyweight', default=0.1, type=float, help='weight for dummy')
parser.add_argument('--optimizer', default='sgd', type=str, help='weight for dummy')
parser.add_argument('--scheduler', default='cos', type=str, help='weight for dummy')
parser.add_argument('--eps', default=1e-3, type=float, help='initialisation for auto-lambda')

opt = parser.parse_args()
epochs = opt.epoch
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
dummy_weight = opt.dummyweight
def compute_encoder_dim(model):
    grad_index = []
    for param in model.encoder_parameter():
        grad_index.append(param.data.numel())
    grad_dim = sum(grad_index)
    return grad_dim

def compute_dummy_dim(model):
    grad_index = []
    for param in model.n_dummy_parameter(0,0):
        grad_index.append(param.data.numel())
    grad_dim = sum(grad_index)
    return grad_dim

def compute_hessian(model, train_data, train_target,n,loss):
    grads_dim = torch.zeros(compute_encoder_dim(model))
    d = torch.autograd.grad(loss,model.n_dummy_parameter(n),retain_graph=True)
    # grad_sq = torch.autograd.grad(loss**2,model.encoder_parameter(),retain_graph=True)
    grad = torch.autograd.grad(loss,model.encoder_parameter(),retain_graph=True)
    norm = torch.cat([w.view(-1) for w in d]).norm(2)
    eps = opt.eps / norm
    # \theta+ = \theta + eps * d_model
    with torch.no_grad():
        for p,g in zip(model.n_dummy_parameter(n), d):
            p += eps *g
    _,train_pred_dummy_list = model(train_data)
    train_pred_dummy = train_pred_dummy_list[n]
    train_loss_dummy = [compute_loss(train_pred_dummy[tn], train_target[task_id],task_id) for tn, task_id in enumerate(train_tasks)]
    dummyloss= sum(train_loss_dummy)
    grad_p = torch.autograd.grad(dummyloss,model.encoder_parameter())
    
    # \theta- = \theta - eps * d_model
    with torch.no_grad():
        for p,g in zip(model.n_dummy_parameter(n), d):
            p -= eps *g
    grad_p_scaled = [g * loss for g in grad_p] 
    grad_scaled = [g * dummyloss for g in grad]
    grad_p_sq = [g * dummyloss for g in grad_p]
    grad_sq = [g *loss for g in grad] 
    prox = [2*((sq+p_sq) - (p_scaled+scaled)) /(opt.eps)**2 for sq,p_sq,p_scaled,scaled in zip(grad_p_sq,grad_sq,grad_p_scaled,grad_scaled)]
    return prox
# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
if opt.with_noise:
    train_tasks = create_task_flags('all', opt.dataset, with_noise=True)
else:
    train_tasks = create_task_flags('all', opt.dataset, with_noise=False)

pri_tasks = create_task_flags(opt.task, opt.dataset, with_noise=False)

train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
print('Dataset: {} | Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode with {}'
      .format(opt.dataset.title(), train_tasks_str, pri_tasks_str, opt.network.upper()))
print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {} + dummy'
      .format(opt.weight.title(), opt.grad_method.upper()))

if opt.weight== 'autol':
    if opt.network == 'split':
        model = MTLDeepLabv3(train_tasks).to(device)
        model.to(device)
    elif opt.network == 'mtan':
        model = MTANDeepLabv3(train_tasks).to(device)
        model.to(device)
else:
    if opt.network == 'split':
        model = MTLDeepLabv3(train_tasks).to(device)
        model = dummy_network_split(model,train_tasks,opt.num_dummy)
        model.to(device)
    elif opt.network == 'mtan':
        model = MTANDeepLabv3(train_tasks).to(device)
        model = dummy_network_mtan(model,train_tasks,opt.num_dummy)
        model.to(device)
total_epoch = epochs
# choose task weighting here
if opt.weight == 'uncert':
    logsigma = torch.tensor([-0.7] * len(train_tasks), requires_grad=True, device=device)
    params = list(model.Nodummy_parameter()) + [logsigma]
    logsigma_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)

if opt.weight in ['dwa', 'equal']:
    T = 2.0  # temperature used in dwa
    lambda_weight = np.ones([total_epoch, len(train_tasks)])
    params = model.Nodummy_parameter()

if opt.weight == 'autol':
    params = model.parameters()
    autol = AutoLambda(model, device, train_tasks, pri_tasks, opt.autol_init)
    meta_weight_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)
    meta_optimizer = optim.Adam([autol.meta_weights], lr=opt.autol_lr)
if opt.grad_method in ['nashmtl', 'imtl']:
    weight_method = AbsWeighting(opt.grad_method, model, task_num=3, device=device)

else:
    pass
if opt.optimizer =='sgd':
    optimizer = optim.SGD(params, lr=0.1, weight_decay=1e-4, momentum=0.9)
else:
    optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-4)
if opt.scheduler == 'cos':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
else:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.5,last_epoch=-1)

# define dataset
if opt.dataset == 'nyuv2':
    dataset_path = 'dataset/nyuv2'
    train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
    test_set = NYUv2(root=dataset_path, train=False)
    batch_size = 4

elif opt.dataset == 'cityscapes':
    dataset_path = 'dataset/cityscapes'
    train_set = CityScapes(root=dataset_path, train=True, augmentation=True)
    test_set = CityScapes(root=dataset_path, train=False)
    batch_size = 4

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)



# a copy of train_loader with different data order, used for Auto-Lambda meta-update
if opt.weight == 'autol':
    val_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)

# apply gradient methods
if opt.grad_method != 'none':
    rng = np.random.default_rng()
    grad_dims = []
    for mm in model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), len(train_tasks)).to(device)




# Train and evaluate multi-task network
train_batch = len(train_loader)
test_batch = len(test_loader)
train_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset)
test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset, include_mtl=True)
test_performed = []
for index in range(total_epoch):
    if index != 0: 
        print(f"{end - start:.5f} sec")

    # apply Dynamic Weight Average
    if opt.weight == 'dwa':
        if index == 0 or index == 1:
            lambda_weight[index, :] = 1.0
        else:
            w = []
            for i, t in enumerate(train_tasks):
                w += [train_metric.metric[t][index - 1, 0] / train_metric.metric[t][index - 2, 0]]
            w = torch.softmax(torch.tensor(w) / T, dim=0)
            lambda_weight[index] = len(train_tasks) * w.numpy()

    # iteration for all batches
    model.train()
    train_dataset = iter(train_loader)
    if opt.weight == 'autol':
        val_dataset = iter(val_loader)
    for k in range(train_batch):
        train_data, train_target = next(train_dataset)
        train_data = train_data.to(device)
        train_target = {task_id: train_target[task_id].to(device) for task_id in train_tasks.keys()}
        if opt.weight== 'autol':      
            val_data, val_target = next(val_dataset)
            val_data = val_data.to(device)
            val_target = {task_id: val_target[task_id].to(device) for task_id in train_tasks.keys()}

            meta_optimizer.zero_grad()

            autol.unrolled_backward(train_data, train_target, val_data, val_target,
                                    scheduler.get_last_lr()[0], optimizer)
            meta_optimizer.step()
            if opt.network == 'split':
                model = dummy_network_split(model,train_tasks,opt.num_dummy)
                model.to(device)
            else:
                model = dummy_network_mtan(model,train_tasks,opt.num_dummy)
                model.to(device)
        else:
            pass
        optimizer.zero_grad()
        train_pred,train_pred_dummy_list = model(train_data)
        train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
        

        train_loss_tmp = [0] * len(train_tasks)
        if opt.weight in ['equal', 'dwa']:
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(lambda_weight[index])]

        if opt.weight == 'uncert':
            train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]

        if opt.weight == 'autol':
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

        # gradient-based methods applied here:
        elif opt.grad_method == "graddrop":
            for i in range(len(train_tasks)):
                train_loss_tmp[i].backward(retain_graph=True)
                grad2vec(model, grads, grad_dims, i)
                model.zero_grad_shared_modules()
            g = graddrop(grads)
            overwrite_grad(model, g, grad_dims, len(train_tasks))
            for n in range(opt.num_dummy):
                train_pred_dummy = train_pred_dummy_list[n]
                train_loss_dummy = [compute_loss(train_pred_dummy[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
                dummyloss = sum(train_loss_dummy)
                prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                for param,pr in zip(model.encoder_parameter(),prox):
                    param.grad += (opt.dummyweight/opt.num_dummy) * pr
            optimizer.step()

        elif opt.grad_method == "pcgrad":
            for i in range(len(train_tasks)):
                train_loss_tmp[i].backward(retain_graph=True)
                grad2vec(model, grads, grad_dims, i)
                model.zero_grad_shared_modules()
            g = pcgrad(grads, rng, len(train_tasks))
            overwrite_grad(model, g, grad_dims, len(train_tasks))
            for n in range(opt.num_dummy):
                train_pred_dummy = train_pred_dummy_list[n]
                train_loss_dummy = [compute_loss(train_pred_dummy[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
                dummyloss = sum(train_loss_dummy)
                prox = compute_hessian(model, train_data, train_target,n,dummyloss)
                for param,pr in zip(model.encoder_parameter(),prox):
                    param.grad += (opt.dummyweight/opt.num_dummy) * pr
            optimizer.step()

        elif opt.grad_method == "cagrad":
            for i in range(len(train_tasks)):
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
        

        if opt.weight == 'autol':
            if opt.network =='split':  
                model = ORIG_network_split(model,train_tasks)
                model.to(device)
            else: 
                model = ORIG_network_mtan(model,train_tasks)
                model.to(device)
        else:
            pass
        train_metric.update_metric(train_pred, train_target, train_loss)

    train_str = train_metric.compute_metric()
    train_metric.reset()
    
    # evaluating test data
    model.eval()
    with torch.no_grad():
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            test_data, test_target = next(test_dataset)
            test_data = test_data.to(device)
            test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}
            if opt.weight == 'autol':
                test_pred = model(test_data)
            else:
                test_pred,_ = model(test_data)
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in
                         enumerate(train_tasks)]

            test_metric.update_metric(test_pred, test_target, test_loss)

    test_str = test_metric.compute_metric()
    test_metric.reset()
    test_performed = test_performed + [test_str]
    scheduler.step()

    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, opt.task.title(), test_metric.get_best_performance(opt.task)))

    if opt.weight == 'autol':
        meta_weight_ls[index] = autol.meta_weights.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': meta_weight_ls}

        print(get_weight_str(meta_weight_ls[index], train_tasks))

    if opt.weight in ['dwa', 'equal']:
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': lambda_weight}

        print(get_weight_str(lambda_weight[index], train_tasks))

    if opt.weight == 'uncert':
        logsigma_ls[index] = logsigma.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': logsigma_ls}

        print(get_weight_str(1 / (2 * np.exp(logsigma_ls[index])), train_tasks))

    np.save('logging/mtl_dense_{}_{}_{}_{}_{}_{}_{}_dummy.npy'
            .format(opt.network, opt.dataset, opt.task, opt.weight, opt.grad_method, opt.seed, opt.dummyweight), dict)
with open('New_result/mtl_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}test_perform_dummy_random{}.txt'.format(opt.network, opt.dataset, opt.task, opt.weight, opt.grad_method, opt.seed, opt.epoch, opt.dummyweight, opt.optimizer,opt.eps,opt.num_dummy),'w',encoding='UTF-8') as f:
    for per in test_performed:
        f.write(per+'\n')

