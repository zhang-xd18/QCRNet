import torch
from torch import nn
import math

NORM = 1

class quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_levels, thrs, levels):
        y = torch.zeros_like(x)
        y_zeros = torch.zeros_like(x)
        for i in range(num_levels - 1):
            g = torch.gt(x,thrs[i])
            y = torch.where(g, y_zeros +levels[i+1], y)
        return y
    @staticmethod
    def backward(self, grad_output):
        return grad_output, None, None, None
    
class Quantization(nn.Module):
    def __init__(self, nbit=6):
        super(Quantization, self).__init__()
        self.nbit = nbit
        self.basis = torch.tensor([(NORM/ (2 ** nbit - 1)) * (2. ** i) for i in range(nbit)])
        self.num_levels = 2 ** nbit

        level_code = []
        for i in range(0,self.num_levels):
            level_code_i = [0. for j in range(nbit)]
            level_num = i
            for j in range(nbit):
                level_code_i[j] = float(level_num % 2)
                level_num = level_num // 2
            level_code.append(level_code_i)
        self.level_code = torch.tensor(level_code)

        thrs_mul = []
        for i in range(1, self.num_levels):
            thrs_mul_i = [0. for j in range(self.num_levels)]
            thrs_mul_i[i-1] = 0.5
            thrs_mul_i[i] = 0.5
            thrs_mul.append(thrs_mul_i)
        self.thrs_mul = torch.tensor(thrs_mul)

        levels = torch.matmul(self.level_code, self.basis)
        self.levels, self.sort_id = torch.sort(levels)
        self.thrs = torch.matmul(self.thrs_mul, levels)

    def forward(self, x):
        q = quant().apply
        xr = x
        x,x_sign = compress(x)
        y = q(x,self.num_levels, self.thrs, self.levels)
        y = decompress(y, x_sign)
        snr = torch.mean(torch.norm(xr, p=2, dim=1)/torch.norm(xr-y, p=2, dim=1))
        return y, snr

def compress(x, mu=50):
    x = 2 * x - 1
    x_sign = torch.where(x > 0, torch.tensor([1],device=x.device), torch.tensor([-1],device=x.device))
    x_c = torch.abs(x)
    x_c = torch.log(torch.ones_like(x_c)+mu*x_c)/math.log(1+mu)
    return x_c, x_sign

def decompress(y,x_sign,mu=50):
    y_d = (torch.exp(y*math.log(1+mu))-1)/mu
    y_d = y_d.mul(x_sign)
    y_d = (y_d + 1)/2
    return y_d

