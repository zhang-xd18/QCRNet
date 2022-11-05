import torch
import torch.nn as nn

from utils.parser import args
from utils import logger, Tester
from utils import init_device, init_model
from dataset import Cost2100DataLoader


def main():
    logger.set_file('./log_LA_{}_{}_{}bit'.format(args.scenario, args.cr, args.nbit))
    
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    logger.info('nbit: {}, scenario: {}, cr: {} '.format(args.nbit, args.scenario,args.cr))

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

    # Define loss function
    # criterion 
    test_criterion = nn.MSELoss().to(device)

    # Inference mode
    if args.evaluate:
        Tester(model, device, test_criterion)(test_data=test_loader, 
            save_data=False, 
            path='./')
        return



if __name__ == "__main__":
    main()
