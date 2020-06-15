#from https://github.com/pytorch/pytorch/pull/1414/files
import math
import torch
from torch.optim import Optimizer


class Nadam(Optimizer):
    """Implements Nadam algorithm.
    It has been proposed in `Incorporating Nesterov Momentum into Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.975, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        schedule_decay (float, optional): beta1 decay factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. _Incorporating Nesterov Momentum into Adam
        https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    """

    def __init__(self, named_params, lr=2e-3, betas=(0.975, 0.999), eps=1e-8,
                 schedule_decay=0, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        schedule_decay=schedule_decay, weight_decay=weight_decay,
                        prod_beta1=1.)
        
        params = []
        
        self.updates = {}
        
        for named_param in named_params:
            dict_ = {
                'name': named_param[0],
                'params': named_param[1]
            }
            if named_param[0] == 'cell.weight_hh_l0':
                self.updates[named_param[0]] = []
            params.append(dict_)
            
        
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                name = group['name']
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                schedule_decay = group['schedule_decay']
                cur_beta1 = beta1 * (1. - 0.5 * (0.96 ** (state['step'] * schedule_decay)))
                next_beta1 = beta1 * (1. - 0.5 * (0.96 ** ((state['step'] + 1) * schedule_decay)))
                prod_beta1 = group['prod_beta1']
                prod_beta1 *= cur_beta1
                next_prod_beta1 = prod_beta1 * next_beta1
                bias_correction1 = (1 - cur_beta1) / (1 - prod_beta1)
                next_bias_correction1 = next_beta1 / (1 - next_prod_beta1)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(cur_beta1).add_(1 - cur_beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                sqrt_bias_correction2 = math.sqrt((1 - beta2 ** state['step']) / beta2)
                step_size = group['lr'] * sqrt_bias_correction2

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                
                if name == 'cell.weight_hh_l0':
                    update = p.data.clone()

                # For memory efficiency, separate update into two
                p.data.addcdiv_(-step_size * next_bias_correction1, exp_avg, denom)
                p.data.addcdiv_(-step_size * bias_correction1, grad, denom)
                
                if name == 'cell.weight_hh_l0':
                    update = p.data - update
                    self.updates[name].append(update)

                # update prod_beta1
                group['prod_beta1'] = prod_beta1

        return loss
    
#https://github.com/pytorch/pytorch/pull/1408/files
#import torch
#from torch.optim import Optimizer
#
#class Nadam(Optimizer):
#    """Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).
#    It has been proposed in `Incorporating Nesterov Momentum into Adam`__.
#    Arguments:
#        params (iterable): iterable of parameters to optimize or dicts defining
#            parameter groups
#        lr (float, optional): learning rate (default: 2e-3)
#        betas (Tuple[float, float], optional): coefficients used for computing
#            running averages of gradient and its square
#        eps (float, optional): term added to the denominator to improve
#            numerical stability (default: 1e-8)
#        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#        schedule_decay (float, optional): momentum schedule decay (default: 4e-3)
#    __ http://cs229.stanford.edu/proj2015/054_report.pdf
#    __ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
#    """
#
#    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
#                 weight_decay=0, schedule_decay=4e-3):
#        defaults = dict(lr=lr, betas=betas, eps=eps,
#                        weight_decay=weight_decay, schedule_decay=schedule_decay)
#        super(Nadam, self).__init__(params, defaults)
#
#    def step(self, closure=None):
#        """Performs a single optimization step.
#        Arguments:
#            closure (callable, optional): A closure that reevaluates the model
#                and returns the loss.
#        """
#        loss = None
#        if closure is not None:
#            loss = closure()
#
#        for group in self.param_groups:
#            for p in group['params']:
#                if p.grad is None:
#                    continue
#                grad = p.grad.data
#                state = self.state[p]
#
#                # State initialization
#                if len(state) == 0:
#                    state['step'] = 0
#                    state['m_schedule'] = 1.
#                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
#                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
#
#                # Warming momentum schedule
#                m_schedule = state['m_schedule']
#                schedule_decay = group['schedule_decay']
#                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                beta1, beta2 = group['betas']
#                eps = group['eps']
#
#                state['step'] += 1
#
#                if group['weight_decay'] != 0:
#                    grad = grad.add(group['weight_decay'], p.data)
#
#                momentum_cache_t = beta1 * \
#                    (1. - 0.5 * (0.96 ** (state['step'] * schedule_decay)))
#                momentum_cache_t_1 = beta1 * \
#                    (1. - 0.5 *
#                     (0.96 ** ((state['step'] + 1) * schedule_decay)))
#                m_schedule_new = m_schedule * momentum_cache_t
#                m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
#                state['m_schedule'] = m_schedule_new
#
#                # Decay the first and second moment running average coefficient
#                bias_correction2 = 1 - beta2 ** state['step']
#
#                exp_avg.mul_(beta1).add_(1 - beta1, grad)
#                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
#                exp_avg_sq_prime = exp_avg_sq.div(1. - bias_correction2)
#
#                denom = exp_avg_sq_prime.sqrt_().add_(group['eps'])
#
#                p.data.addcdiv_(-group['lr'] * (1. - momentum_cache_t) / (1. - m_schedule_new), grad, denom)
#                p.data.addcdiv_(-group['lr'] * momentum_cache_t_1 / (1. - m_schedule_next), exp_avg, denom)
#
#        return loss