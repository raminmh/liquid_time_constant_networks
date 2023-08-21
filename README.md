# Liquid time-constant Networks (LTCs)

[Update] A Pytorch version together with tutorials are added to our sister repository: 
[https://github.com/mlech26l/ncps](https://github.com/mlech26l/ncps)

This is the official repository for LTC networks described in the paper: https://arxiv.org/abs/2006.04439
This repository allows you to train continuous-time models with backpropagation through-time (BPTT). Available Continuous-time models are: 
| Models | References |
| ----- | ----- |
| Liquid time-constant Networks | https://arxiv.org/abs/2006.04439 |
| Neural ODEs | https://papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf |
| Continuous-time RNNs | https://www.sciencedirect.com/science/article/abs/pii/S089360800580125X |
Continuous-time Gated Recurrent Units (GRU) | https://arxiv.org/abs/1710.04110 |

## Requisites

All models were implemented and tested with TensorFlow 1.14.0 and python3 on Ubuntu 16.04 and 18.04 machines.
All the following steps assume that they are executed under these conditions.

## Preparation

First, we have to download all datasets by running 
```bash
source download_datasets.sh
```
This script creates a folder ```data```, where all downloaded datasets are stored.

## Training and evaluating the models 

There is exactly one Python module per dataset:
- Hand gesture segmentation: ```gesture.py```
- Room occupancy detection: ```occupancy.py```
- Human activity recognition: ```har.py```
- Traffic volume prediction: ```traffic.py```
- Ozone level forecasting: ```ozone.py```

Each script accepts the following four arguments:
- ```--model: lstm | ctrnn | ltc | ltc_rk | ltc_ex```
- ```--epochs: number of training epochs (default 200)```
- ```--size: number of hidden RNN units  (default 32)```
- ```--log: interval of how often to evaluate validation metric (default 1)```

Each script trains the specified model for the given number of epochs and evaluates the
validation performance after every ``log`` steps.
At the end of the training, the best-performing checkpoint is restored and the model is evaluated on the test set.
All results are stored in the ```results``` folder by appending the result to CSV file.

For example, we can train and evaluate the CT-RNN by executing
```bash
python3 har.py --model ctrnn
```
After the script is finished there should be a file ```results/har/ctrnn_32.csv``` created, containing the following columns:
- ```best epoch```: Epoch number that achieved the best validation metric
- ```train loss```: Training loss achieved at the best epoch
- ```train accuracy```: Training metric achieved at the best epoch
- ```valid loss```: Validation loss achieved at the best epoch
- ```valid accuracy```: Best validation metric achieved during training
- ```test loss```: Loss on the test set
- ```test accuracy```: Metric on the test set

## Hyperparameters

| Parameter | Value | Description | 
| ---- | ---- | ------ |
| Minibatch size | 16 | Number of training samples over which the gradient descent update is computed |
| Learning rate | 0.001/0.02 | 0.01-0.02 for LTC, 0.001 for all other models. |
| Hidden units | 32 | Number of hidden units of each model |
| Optimizer | Adam | See (Kingma and Ba, 2014) |
| beta_1 | 0.9 | Parameter of the Adam method |
| beta_2 | 0.999 | Parameter of the Adam method |
| epsilon | 1e-08 | Epsilon-hat parameter of the Adam method |
| Number of epochs | 200 | Maximum number of training epochs |
| BPTT length | 32 | Backpropagation through time length in time-steps | 
| ODE solver sreps | 1/6 | relative to input sampling period |
| Validation evaluation interval | 1 | Interval of training epochs when the metrics on the validation are evaluated  | 


# Trajectory Length Analysis

Run the ```main.m``` file to get trajectory length results for the desired setting tuneable in the code. 


