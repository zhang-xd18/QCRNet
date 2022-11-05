import time
import torch
import math
from collections import namedtuple

from utils import logger
from utils.statics import AverageMeter, evaluator

__all__ = ['Tester']


field = ('nmse', 'rho', 'epoch','SNR')
Result = namedtuple('Result', field, defaults=(None,) * len(field))

class Tester:
    r""" The testing interface for classification
    """
    def __init__(self, model, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq

    def __call__(self, test_data, verbose=True):
        r""" Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            loss, nmse, SNR = self._iteration(test_data)
        if verbose:
            logger.info(f'\n=> Test result: \nloss: {loss:.3e}'
                  f'  NMSE: {nmse:.3e}'
                  f'  SNR: {20*math.log10(SNR): .3e}\n')
        return loss, nmse, SNR

    def _iteration(self, data_loader):
        r""" protected function which test the model on given data loader for one epoch.
        """

        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        iter_SNR = AverageMeter('iter_SNR')
        time_tmp = time.time()

        for batch_idx, (sparse_gt, ) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred, codeword, codewordQ, SNR = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt, codeword, codewordQ)
            nmse = evaluator(sparse_pred, sparse_gt)

            # Log and visdom update
            iter_loss.update(loss)
            iter_nmse.update(nmse)
            iter_time.update(time.time() - time_tmp)
            iter_SNR.update(SNR)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'loss: {iter_loss.avg:.3e} | '
                            f'NMSE: {iter_nmse.avg:.3e} | SNR: {20*math.log10(iter_SNR.avg): .3e} | time: {iter_time.avg:.3f}')

        logger.info(f'=> Test NMSE: {iter_nmse.avg:.3e}\n')

        return iter_loss.avg, iter_nmse.avg, iter_SNR.avg
