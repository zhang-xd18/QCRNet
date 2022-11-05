import torch
import torch.nn as nn

from utils.parser import args
from utils import logger, Tester
from utils import init_device, init_model
from dataset import Cost2100DataLoader
import math

class Loss(nn.Module):
    def __init__(self, mode='const', lamda_start=1e-2, lamda_end=0, lamda_max=5e-2, T_max=1000, alpha=0):
        super(Loss, self).__init__()
        self.mode = mode
        self.lamda_start = lamda_start
        self.lamda_max = lamda_max
        self.lamda_end = lamda_end
        self.T_max = T_max
        self.lamda = self.lamda_start
        self.NMSE_loss = None
        self.QSNR_loss = None
        self.alpha = alpha

    def forward(self, pred, label, codeword, codewordQ):
        mse1 = nn.MSELoss()
        mse2 = nn.MSELoss()
        norm = torch.norm(codeword - 0.5,p=1)/(codeword.numel()*0.5)
        loss = mse1(pred, label) + self.lamda*mse2(codeword, codewordQ) + self.alpha*norm
        self.NMSE_loss = mse1(pred, label)
        self.QSNR_loss = mse2(codeword, codewordQ)
        return loss

    def step(self, ep):
        if self.mode == 'linear':
            if ep <= self.T_max/2:
                self.lamda = self.lamda_start + (ep - 1)*(self.lamda_max - self.lamda_start)/(self.T_max/2 - 1)
            else:
                self.lamda = self.lamda_max + (ep - self.T_max/2)*(self.lamda_end - self.lamda_max)/(self.T_max/2)
        elif self.mode == 'const':
            return

    def get_lamda(self):
        return self.lamda

    def get_loss(self):
        return self.NMSE_loss, self.QSNR_loss


def main():
    logger.set_file('./log_NA_{}_{}_{}bit'.format(args.scenario, args.cr, args.nbit))
    
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    if args.scenario == 'in':
        bottle = 'b'
    else:
        bottle = 'pb'
    logger.info('nbit: {}, scenario: {}, cr: {}, bottle: {}'.format(args.nbit, args.scenario, args.cr, bottle))

    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the data loader
    _, _, test_loader = Cost2100DataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        scenario=args.scenario,
        device=device)()

    # Define model
    model = init_model(args)
    model.to(device)

    # Inference mode
    if args.evaluate:
        criterion = Loss(mode='const',lamda_start=0)
        Tester(model, device, criterion)(test_data=test_loader)
        return
          
if __name__ == "__main__":
    main()
