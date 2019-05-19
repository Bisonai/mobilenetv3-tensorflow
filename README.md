# MobileNetV3 TensorFlow
Unofficial implementation of MobileNetV3 architecture described in paper [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244).
This repository contains [small](https://github.com/bisonai/mobilenetv3-tensorflow/blob/master/mobilenetv3_small.py) and [large](https://github.com/bisonai/mobilenetv3-tensorflow/blob/master/mobilenetv3_large.py) MobileNetV3 architecture implemented using TensforFlow with `tf.keras` API.

## Requirements
* Python 3.6+
* TensorFlow 1.13+

```shell
pip install -r requirements.txt
```


## Build model

### MobileNetV3 Small
```python
from mobilenetv3_factory import build_mobilenetv3
model = build_mobilenetv3(
    "small",
    input_shape=(224, 224, 3),
    num_classes=1001,
    width_multiplier=1.0,
)
```

### MobileNetV3 Large

```python
from mobilenetv3_factory import build_mobilenetv3
model = build_mobilenetv3(
    "large",
    input_shape=(224, 224, 3),
    num_classes=1001,
    width_multiplier=1.0,
)
```

## Train

### CIFAR10 dataset

```shell
python train.py \
    --model_type small \
    --width_multiplier 1.0 \
    --num_classes 10 \
    --height 96 \
    --width 96 \
    --channel 3 \
    --dataset cifar10 \
    --lr 0.01 \
    --optimizer rmsprop \
    --train_batch_size 256 \
    --valid_batch_size 256 \
    --num_epoch 10 \
    --logdir logdir
```

### MNIST dataset

```shell
python train.py \
    --model_type small \
    --width_multiplier 1.0 \
    --num_classes 10 \
    --height 96 \
    --width 96 \
    --channel 1 \
    --dataset mnist \
    --lr 0.01 \
    --optimizer rmsprop \
    --train_batch_size 256 \
    --valid_batch_size 256 \
    --num_epoch 10 \
    --logdir logdir
```

## Evaluate

### CIFAR10 dataset

```shell
python evaluate.py \
    --model_type small \
    --width_multiplier 1.0 \
    --num_classes 10 \
    --height 96 \
    --width 96 \
    --channel 3 \
    --dataset cifar10 \
    --valid_batch_size 256 \
    --model_path mobilenetv3_small_cifar10_10.h5
```

### MNIST dataset

```shell
python evaluate.py \
    --model_type small \
    --width_multiplier 1.0 \
    --num_classes 10 \
    --height 96 \
    --width 96 \
    --channel 1 \
    --dataset mnist \
    --valid_batch_size 256 \
    --model_path mobilenetv3_small_mnist_10.h5
```

## TensorBoard
Graph, training and evaluaion metrics are saved to TensorBoard event file uder directory specified with --logdir` argument during training.
You can launch TensorBoard using following command.

```shell
tensorboard --logdir logdir
```


## License
[Apache License 2.0](https://github.com/bisonai/mobilenetv3-tensorflow/blob/master/LICENSE)
