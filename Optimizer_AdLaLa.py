import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer,required

class AdLaLa(Optimizer):
    def __init__(self,params,lr=0.25,eps=0.05,sigA = 5e-4,T1=1e-6,cgamma=0,dgamma=0):
        defaults = dict(lr=lr,eps=eps,sigA=sigA,T1=T1,cgamma=cgamma,dgamma=dgamma)
        super(AdLaLa,self).__init__(params,defaults)
           
    def __setstate__(self,state):
        super(AdLaLa,self).__setstate__(state)

    @torch.no_grad()
    def stepMom(self): 
        for group in self.param_groups: 
            num = 0
            lr = group['lr'] # Stepsize lr 

            for p in group['params']: # Runs through the network parameters

                param_state = self.state[p]

                d_p = p.grad  # Compute gradients (derivative of the loss wrt the parameters)

                # Initialize the momenta to be zero 
                Mom = param_state['momentum_buffer'] = 0*torch.clone(d_p).detach()

                # Take a half B step: update momenta Mom = Mom - (lr/2)*Grad, where grad = d_p, lr = stepsize
                Mom.add_(-lr*d_p/2) 

                if num < 2:  # If first layer, then we want to use the Adaptive Langevin optimizer
                    # Initialize xi variable 
                    if num == 0: # For the weights
                        xi = param_state['xis'] = 0.1*torch.ones_like(d_p[0][0]).detach()
                    else: # For the biases
                        xi = param_state['xis'] = 0.1*torch.ones_like(d_p[0]).detach()

                num += 1
                

    @torch.no_grad()
    def stepAOA(self):

        for group in self.param_groups:
            num = 0

            lr = group['lr'] # stepsize
            cgamma = group['cgamma']
            dgamma = group['dgamma']
            eps = group['eps']
            sigA  = group['sigA']
            T1 = group['T1']

            for p in group['params']: # Runs through the networks parameters
                param_state = self.state[p]

                if num < 2: # If first layer, then we want to use the Adaptive Langevin optimizer
                # For Adaptive langevin we use the ACDEDCAB optimizer 
                # The A and B steps are the same as for the Langevin BAOAB scheme,
                # for more details see p.10 in our arXiv 1908.11843 paper
                    
                    # Recall the current state of the momenta and xi variables
                    Mom = param_state['momentum_buffer'] 
                    xi = param_state['xis']  
                    
                    # A-step:  update parameters p = p + (lr/2)*Mom, with stepsize lr
                    p.add_(Mom*lr/2)

                    # C-step: update momenta Mom = exp(-xi*lr/2)*Mom
                    C = torch.exp(-xi*lr/2)
                    Mom.mul_(C)

                    # D-step: update momenta Mom = Mom + sig*sqrt(lr/2)*R,  
                    # where R is a standard normal random vector with iid components
                    shapep = p.size()
                    D = sigA*np.sqrt(lr/2)
                    Mom.add_(D*torch.cuda.FloatTensor(*shapep).normal_())
                    
                    # E-step:update xi = xi + eps*lr*(Mom^T Mom - N*T1), where N = # of degrees of freedom
                    if num == 0: # For the weights
                        E = torch.dot(Mom.reshape(shapep[0]*shapep[1],1).squeeze(),Mom.reshape(shapep[0]*shapep[1],1).squeeze())-shapep[0]*shapep[1]*T1
                    else: # For the biases
                        E = torch.dot(Mom.reshape(shapep[0],1).squeeze(),Mom.reshape(shapep[0],1).squeeze())-shapep[0]*T1
                   
                    F = eps*lr
                    xi.add_(F*E) 

                    # D-step: update momenta Mom = Mom + D*R, D = sig*sqrt(lr/2)
                    Mom.add_(D*torch.cuda.FloatTensor(*shapep).normal_()) 
                    
                    # C-step: update momenta Mom = exp(-xi*lr/2)*Mom
                    C = torch.exp(-xi*lr/2)
                    Mom.mul_(C)
                    
                    # A-step: update parameters p = p + (lr/2)*Mom
                    p.add_(Mom*lr/2)

                else:  # If not first layer, then we want to use the Langevin optimizer
                # For langevin we use the BAOAB optimizer 
                # -> in practice we use AOAB, for more details see p.7 in our arXiv 1908.11843 paper
                    
                    Mom = param_state['momentum_buffer'] # Recall the current state of the momenta variables
                    
                    # A-step: update parameters p = p + (lr/2)*Mom, with stepsize lr
                    p.add_(Mom*lr/2)
                    
                    # O-step:  update momenta Mom = cgam*Mom + dgam*R,
                    # where R is a standard normal random vector with iid components and cgam and dgam are defined in main file
                    shapep = p.size()
                    Mom.mul_(cgamma).add_(torch.cuda.FloatTensor(*shapep).normal_(),alpha=dgamma)

                    # A-step:  update parameters p = p + (lr/2)*Mom,
                    p.add_(Mom*lr/2)

                num += 1

    @torch.no_grad()
    def stepB(self):

         for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:

                if p.grad is None:
                    continue
                
                # Compute gradients (derivative of the loss wrt the parameters)
                d_p = p.grad 

                param_state = self.state[p]
                
                # Recall the current state of the momenta variables
                Mom = param_state['momentum_buffer'] 
                
                # B-step: update momenta Mom = Mom - lr*grad, where grad = d_p, lr = stepsize
                Mom.add_(-lr*d_p)

                
               
