# -*- coding:utf-8 -*-
"""Static MSTGCN baseline retained for compatibility with the original project."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial


class cheb_conv(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super().__init__(); self.K=K; self.out_channels=out_channels
        self.register_buffer('cheb_stack', torch.stack([p.float() for p in cheb_polynomials], dim=0))
        self.Theta=nn.ParameterList([nn.Parameter(torch.empty(in_channels,out_channels)) for _ in range(K)])
    def forward(self,x):
        B,N,_,T=x.shape; ys=[]
        for t in range(T):
            sig=x[:,:,:,t]; out=torch.zeros(B,N,self.out_channels,device=x.device)
            for k in range(self.K):
                rhs=sig.permute(0,2,1).matmul(self.cheb_stack[k].to(x.device)).permute(0,2,1)
                out=out+rhs.matmul(self.Theta[k])
            ys.append(out.unsqueeze(-1))
        return F.relu(torch.cat(ys,dim=-1))


class MSTGCN_block(nn.Module):
    def __init__(self,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,cheb_polynomials):
        super().__init__(); self.cheb_conv=cheb_conv(K,cheb_polynomials,in_channels,nb_chev_filter)
        self.time_conv=nn.Conv2d(nb_chev_filter,nb_time_filter,(1,3),stride=(1,time_strides),padding=(0,1))
        self.residual_conv=nn.Conv2d(in_channels,nb_time_filter,(1,1),stride=(1,time_strides)); self.ln=nn.LayerNorm(nb_time_filter)
    def forward(self,x):
        s=self.cheb_conv(x); tc=self.time_conv(s.permute(0,2,1,3)); r=self.residual_conv(x.permute(0,2,1,3))
        return self.ln(F.relu(r+tc).permute(0,3,2,1)).permute(0,2,3,1)


class MSTGCN_submodule(nn.Module):
    def __init__(self,DEVICE,nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,cheb_polynomials,num_for_predict,len_input):
        super().__init__(); self.BlockList=nn.ModuleList([MSTGCN_block(in_channels,K,nb_chev_filter,nb_time_filter,time_strides,cheb_polynomials)])
        self.BlockList.extend([MSTGCN_block(nb_time_filter,K,nb_chev_filter,nb_time_filter,1,cheb_polynomials) for _ in range(nb_block-1)])
        self.final_conv=nn.Conv2d(int(len_input/time_strides),num_for_predict,kernel_size=(1,nb_time_filter)); self.to(DEVICE)
    def forward(self,x):
        for block in self.BlockList: x=block(x)
        return self.final_conv(x.permute(0,3,1,2))[:,:,:,-1].permute(0,2,1)


def make_model(DEVICE,nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,adj_mx,num_for_predict,len_input):
    L=scaled_Laplacian(adj_mx); cheb=[torch.from_numpy(i).float().to(DEVICE) for i in cheb_polynomial(L,K)]
    model=MSTGCN_submodule(DEVICE,nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,cheb,num_for_predict,len_input)
    for p in model.parameters():
        if p.dim()>1: nn.init.xavier_uniform_(p)
    return model
