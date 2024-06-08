import multiprocessing

import numpy as np
import torch

import cerebras.pytorch.distributed as dist


class NumpyMmapDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, sequence_length):
        self.sequence_length = sequence_length
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")

    def __getitem__(self, i):
        x = self.data[
            i * self.sequence_length : (i + 1) * self.sequence_length + 1
        ]
        x = x.astype(np.int32)
        return x[:-1], x[1:]  # input_ids, labels

    def __len__(self):
        return (len(self.data) - 1) // self.sequence_length


class Sharder(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.task_id = dist.get_streaming_rank() if dist.is_streamer() else 0
        self.num_tasks = dist.num_streamers() if dist.is_streamer() else 1

    def __iter__(self):
        for i, x in enumerate(self.data_source):
            if i % self.num_tasks == self.task_id:
                yield x

    def __len__(self):
        l = len(self.data_source)
        return l // self.num_tasks + 1 * (self.task_id < l % self.num_tasks)


def get_dataloader(data_path, sequence_length, batch_size, seed=0):
    gen = torch.Generator()
    gen.manual_seed(seed)
    dataset = NumpyMmapDataset(data_path, sequence_length)
    sampler = torch.utils.data.RandomSampler(dataset, generator=gen)
    sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
    sampler = Sharder(sampler)
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=min(multiprocessing.cpu_count(), 8),
        persistent_workers=True,
    )