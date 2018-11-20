
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed

cudnn.benchmark = True

from torchvision.models import resnet50


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Minimal Distributed')
    parser.add_argument('--distributed', action='store_true', help='Run distributed training. Default True')
    parser.add_argument('--deprecated', action='store_true', help='Use old distributed code')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    return parser

cudnn.benchmark = True
args = get_parser().parse_args()

if args.deprecated:
  import torch.distributed.deprecated as dist
  from torch.nn.parallel.deprecated import DistributedDataParallel
else:
  import torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel

def env_world_size(): return int(os.environ['WORLD_SIZE'])
def env_rank(): return int(os.environ['RANK'])

def sum_tensor(tensor):
  rt = tensor.clone()
  dist.all_reduce(rt, op=dist.reduce_op.SUM)
  return rt


def BN_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    return BN_convert_float(network.half())



def main():
    print('Distributed initializing process group')
    torch.cuda.set_device(args.local_rank)
    if args.deprecated: dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=env_world_size())
    else: dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=env_world_size(), rank=env_rank())
    assert(env_world_size() == dist.get_world_size())
    print("Distributed: success (%d/%d)"%(args.local_rank, dist.get_world_size()))

    print('Loading model')
    model = resnet50().cuda()

    print('Network to fp16')
    model = network_to_half(model)

    print('1 Deadlock may happen here')
    tensor = torch.tensor([1.0]).float().cuda()
    print('1 Creating tensor:', tensor.item())
    output = sum_tensor(tensor)
    print('1 Able to sync machines:', output.item())


    print('Loading distributed')
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    print('Distributed model loaded')

    print('Deadlock may happen here')
    tensor = torch.tensor([1.0]).float().cuda()
    print('Creating tensor:', tensor.item())
    output = sum_tensor(tensor)
    print('Able to sync machines:', output.item())

    print('DONE')

if __name__ == '__main__':
    main()


