import torch, sys, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import cvxpy as cp
except ModuleNotFoundError:
    from pip._internal import main as pip
    pip(['install', '--user', 'cvxpy'])
    import cvxpy as cp




class AbsWeighting:
    def __init__(self, model,task_num,device):
        super().__init__()
        self.model=model
        self.task_num=task_num
        self.device = device
    def init_param(self):
        
        pass

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.model.encoder_parameter():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.model.encoder_parameter():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode):
        '''
        mode: backward, autograd
        '''

        grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        for tn in range(self.task_num):
            if mode == 'backward':
                losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward(retain_graph=True)
                grads[tn] = self._grad2vec()
            elif mode == 'autograd':
                grad = list(torch.autograd.grad(losses[tn], self.model.encoder_parameter(), retain_graph=True))
                grads[tn] = torch.cat([g.view(-1) for g in grad])
            else:
                raise ValueError('No support {} mode for gradient computation')
            self.model.zero_grad_shared_modules()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.model.encoder_parameter():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
            
    def _get_grads(self, losses, mode='backward'):
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode)
        return grads
    
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
        self._reset_grad(new_grads)
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass



class IMTL(AbsWeighting, nn.Module):
    def __init__(self,model,task_num,device):
        super().__init__(model,task_num,device)
        self.loss_scale = nn.Parameter(torch.tensor([0.0]*self.task_num, device=self.device))
        
    def backward(self, losses, **kwargs):
        losses =  [w.exp() * losses[i] -w for i, w in enumerate(self.loss_scale)]
        grads = self._get_grads(losses, mode='backward')
        
        grads_unit = grads/torch.norm(grads, p=2, dim=-1, keepdim=True)

        D = grads[0:1].repeat(self.task_num-1, 1) - grads[1:]
        U = grads_unit[0:1].repeat(self.task_num-1, 1) - grads_unit[1:]

        alpha = torch.matmul(torch.matmul(grads[0], U.t()), torch.inverse(torch.matmul(D, U.t())))
        alpha = torch.cat((1-alpha.sum().unsqueeze(0), alpha), dim=0)
        self._backward_new_grads(alpha, grads=grads)
        return alpha.detach().cpu().numpy()


class Nash_MTL(AbsWeighting):
    r"""Nash-MTL.
    
    This method is proposed in `Multi-Task Learning as a Bargaining Game (ICML 2022) <https://proceedings.mlr.press/v162/navon22a/navon22a.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/AvivNavon/nash-mtl>`_. 

    Args:
        update_weights_every (int, default=1): Period of weights update.
        optim_niter (int, default=20): The max iteration of optimization solver.
        max_norm (float, default=1.0): The max norm of the gradients.


    .. warning::
            Nash_MTL is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self,model,task_num,device):
        super().__init__(model,task_num,device)
        
        self.step = 0
        self.prvs_alpha_param = None
        self.init_gtg = np.eye(self.task_num)
        self.prvs_alpha = np.ones(self.task_num, dtype=np.float32)
        self.normalization_factor = np.ones((1,))
        
    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )
        
    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha
        
    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.task_num,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.task_num,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.task_num, self.task_num), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.task_num):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)
        
    def backward(self, losses, **kwargs):
        self.update_weights_every =1
        self.optim_niter = 20
        self.max_norm = 1
        
        if self.step == 0:
            self._init_optim_problem()
        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='autograd')
            
            GTG = torch.mm(grads, grads.t())
            self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha).to(torch.float32).to(self.device)
        else:
            self.step += 1
            alpha = self.prvs_alpha
        train_loss_tmp = [w * losses[i] for i, w in enumerate(alpha)]
        train_loss = sum(train_loss_tmp)
        train_loss.backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.encoder_parameter(), self.max_norm)
        
        return alpha.detach().cpu().numpy()

class AbsWeighting:
    def __init__(self, method: str, model, task_num: int, device: torch.device):
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](model, task_num=task_num, device=device)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
        self, losses, **kwargs):
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.model.parameters()

METHODS = dict(
    nashmtl=Nash_MTL,
    imtl=IMTL
)
