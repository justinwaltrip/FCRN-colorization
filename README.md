# Effects of Colorization Pretraining on Depth Estimation

## Installation

```
chmod +x setup.sh
./setup.sh
```

To setup wandb, run:

```
wandb login
```

## Data

To download and process the NYUv2 dataset, run:

```
mkdir -p data/nyu
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat -P data/nyu
python dataloaders/nyu_convert.py
```

## Baseline

To overfit a baseline depth model on NYUv2, run:

```
python main.py \
    --dataset nyu \
    --batch-size 16 \
    --epochs 200 \
    --lr 0.00001 \
    --lr_patience 10 \
    --momentum 0.9 \
    --weight_decay 0 \
    --overfit
```

To train a baseline depth model on NYUv2, run:

```
python main.py \
    --dataset nyu \
    --batch-size 32 \
    --epochs 200 \
    --lr 0.00001 \
    --lr_patience 10 \
    --momentum 0.9 \
    --weight_decay 0
```

## Colorization Pretraining

To overfit a pretraining model on NYUv2, run:

```
python pretrain.py \
    --dataset nyu \
    --batch-size 16 \
    --epochs 200 \
    --lr 0.0001 \
    --lr_patience 10 \
    --momentum 0.9 \
    --weight_decay 0 \
    --overfit
```
