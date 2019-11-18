import torch
from torch.utils.data.dataset import Dataset


def one_batch_dataset(dataset, batch_size):
    print("==> Grabbing a single batch")

    perm = torch.randperm(len(dataset))

    one_batch = [dataset[idx.item()] for idx in perm[:batch_size]]

    class _OneBatchWrapper(Dataset):
        def __init__(self):
            self.batch = one_batch

        def __getitem__(self, index):
            return self.batch[index]

        def __len__(self):
            return len(self.batch)

    return _OneBatchWrapper()
