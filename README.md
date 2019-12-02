# What's hidden in a randomly weighted neural network?

by Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari

arxiv link: https://arxiv.org/abs/1911.13299
<!-- ![alt text](images/teaser.png) -->

<p align="center">
<img width="700" src="images/teaser.png">
</p>

## Setup

1. Set up a virtualenv with python 3.7.4. You can use pyvenv or conda for this.
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
               --name example \
               --data <path/to/data-dir> \
               --prune-rate 0.5
```

### Tracking

```
tensorboard --logdir runs/ --bind_all
```

When your experiment is done, a CSV entry will be written (or appended) to ```runs/results.csv```. Your experiment base directory will automatically be written to ```runs/<config-name>/prune-rate=<prune-rate>/<experiment-name>``` with ```checkpoints/``` and ```logs/``` subdirectories. If your experiment happens to match a previously created experiment base directory then an integer increment will be added to the filepath (eg. ```/0```, ```/1```, etc.). Checkpoints by default will have the first, best, and last models. To change this behavior, use the ```--save-every``` flag. 

## Requirements

Python 3.7.4, CUDA Version 10.1 (also works with 9.2 and 10.0):

```
absl-py==0.8.1
grpcio==1.24.3
Markdown==3.1.1
numpy==1.17.3
Pillow==6.2.1
protobuf==3.10.0
PyYAML==5.1.2
six==1.12.0
tensorboard==2.0.0
torch==1.3.0
torchvision==0.4.1
tqdm==4.36.1
Werkzeug==0.16.0
```
