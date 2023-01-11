import os
import torch as th
from transformers import BertConfig, BertModel
import ray
import argparse


@ray.remote(num_cpus=1, num_gpus=1)
def gpu_heating():
    config = BertConfig()
    batch_size = 16
    model = BertModel(config).to("cuda")
    input_ids = th.randint(0, 100, (batch_size, 128)).to("cuda")
    attention_mask = th.ones_like(input_ids).to("cuda")
    device_name = th.cuda.get_device_name()
    # get environment variable CUDA_VISIBLE_DEVICES
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = 0
    while True:
        # forward pass
        out = model(input_ids, attention_mask=attention_mask)
        if count % 100 == 0:
            print("GPU {}:{} Finished {} iterations".format(device_name, visible_devices, count * batch_size))
        count += 1


@ray.remote(num_cpus=1)
def cpu_heating(cpu_id):
    th.set_num_threads(1)
    config = BertConfig()
    batch_size = 1
    model = BertModel(config)
    input_ids = th.randint(0, 100, (batch_size, 128))
    attention_mask = th.ones_like(input_ids)
    count = 0
    # print GPU id
    while True:
        # forward pass
        model(input_ids, attention_mask=attention_mask)
        if count % 100 == 0:
            print("CPU {} Finished {} iterations".format(cpu_id, count * batch_size))
        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=2)
    parser.add_argument('--num-cpus', type=int, default=16)
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)

    # start heating
    res = []
    for _ in range(args.num_gpus):
        r = gpu_heating.remote()
        res.append(r)
    for i in range(args.num_cpus):
        r = cpu_heating.remote(i)
        res.append(r)

    ray.get(res)
