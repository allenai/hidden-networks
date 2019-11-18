import os
import torch
from torchvision import datasets, transforms


class MNIST:
    def __init__(self, args):
        super(MNIST, self).__init__()

        data_root = os.path.join(args.data, "mnist")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
