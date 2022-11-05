import argparse

parser = argparse.ArgumentParser(description='CRNet PyTorch Training')


# ========================== Indispensable arguments ==========================

parser.add_argument('--data-dir', type=str, required=True,
                    help='the path of dataset.')
parser.add_argument('--scenario', type=str, required=True, choices=["in", "out", "Q","A30","A300"],
                    help="the channel scenario")
parser.add_argument('-b', '--batch-size', type=int, required=True, metavar='N',
                    help='mini-batch size')
parser.add_argument('-j', '--workers', type=int, metavar='N', required=True,
                    help='number of data loading workers')


# ============================= Optical arguments =============================

# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu-affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio')
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')

# parser.add_argument('--lamda', type=float, default=0.1,
#                     help='normalization rate')

parser.add_argument('--nbit', type=int, default=8,
                    help='quantization bit number')

# parser.add_argument('--mode', type=str, default='mu',
#                     help='quantization mode')

# parser.add_argument('--save_data', type=bool, default=False,
#                     help='Save the data processing')


args = parser.parse_args()
