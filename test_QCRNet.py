import torch
import torch.nn as nn

from utils.parser import args
from utils import logger, Tester
from utils import init_device, init_model
from dataset import Cost2100DataLoader

def main():
    logger.set_file('./log.out')
    
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    logger.info('scenario: {}, cr: {}, nbit: {}'.format(args.scenario, args.cr, args.nbit))

    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the data loader
    test_loader = Cost2100DataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        scenario=args.scenario)()

    # Define model
    model = init_model(args)
    model.to(device)
    
    criterion = nn.MSELoss().to(device)

    # Inference mode
    if args.evaluate:
        Tester(model, device, criterion)(test_loader)
        return

if __name__ == "__main__":
    main()
