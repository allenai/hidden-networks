# Code release for "What's hidden in a randomly weighted neural network?"

by Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari

<!-- ![alt text](images/teaser.png) -->

<p align="center">
<img width="700" src="images/teaser.png">
</p>

## Setup

1. Set up a virtualenv with python 3.7.4
2. Run ```pip install -r requirements.txt``` to get requirements
3. Create a data directory as a base for all datasets. For example, if your base directory is ```/mnt/datasets``` then imagenet would be located at ```/mnt/datasets/imagenet``` and CIFAR-10 would be located at ```/mnt/datasets/cifar10```

## Starting an Experiment

We use config files located in the ```configs/``` folder to organize our experiments. The basic setup for any experiment is:

```bash
python main.py --config <path/to/config> <override-args>
```

Common example ```override-args``` include ```--multigpu=<gpu-ids seperated by commas, no spaces>``` to run on GPUs, and ```--prune-rate``` to set the prune rate ```(1 - weights_remaining)``` for an experiment. Run ```python main --help``` for more details.

### YAML Name Key

```
(u)uc -> (unscaled) unsigned constant
(u)sc -> (unscaled) signed constant
(u)pt -> (unscaled) pretrained init
(u)kn -> (unscaled) kaiming normal
```

### Example Run

```bash
python main.py --config configs/smallscale/conv4/conv4_usc_unsigned.yml \
               --multigpu 0 \
               --name maskonly \
               --data <path/to/dataset> \
               --prune-rate 0.5
```

