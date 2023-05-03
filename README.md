# Effects of Colorization Pretraining on Depth Estimation

## Installation

```
chmod +x setup.sh
./setup.sh
```

## Data

To download and process the NYUv2 dataset, run:

```
mkdir -p data/nyu
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat -P data/nyu
python dataloaders/nyu_convert.py
```

## Baseline

To train a baseline depth model on NYUv2, run:

```
python main.py --dataset nyu
```
