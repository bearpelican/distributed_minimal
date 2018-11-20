#!/usr/bin/env python

import argparse
import ncluster
import os

IMAGE_NAME = 'pytorch.imagenet.source.v9.c10d'
INSTANCE_TYPE = 'p3.8xlarge'
NUM_GPUS = {'p3.2xlarge': 1, 'p3.8xlarge':4, 'p3.16xlarge':8}[INSTANCE_TYPE]

ncluster.set_backend('aws')
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='distributed_minimal',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=2,
                    help="how many machines to use")
args = parser.parse_args()

def main():
  supported_regions = ['us-east-1']
  assert ncluster.get_region() in supported_regions, f"required AMI {IMAGE_NAME} has only been made available in regions {supported_regions}, but your current region is {ncluster.get_region()}"

  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          num_tasks=args.machines,
                          image_name=IMAGE_NAME,
                          instance_type=INSTANCE_TYPE
                          )
  job.upload('training/train_minimal.py')
  job.run(f'conda activate pytorch_source')

  nccl_params = 'CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL'

  # TODO: simplify args processing, or give link to actual commands run
  for i, task in enumerate(job.tasks):
    dist_params = f'--nproc_per_node={NUM_GPUS} --nnodes={args.machines} --node_rank={i} --master_addr={job.tasks[0].ip} --master_port={6006}'
    cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} train_minimal.py'
    task.run(cmd, non_blocking=True)


if __name__ == '__main__':
  main()
