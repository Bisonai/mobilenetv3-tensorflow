# MobileNetV3 TensorFlow
Unofficial implementation of MobileNetV3 architecture described in paper [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244).
This repository contains [small](https://github.com/bisonai/mobilenetv3-tensorflow/blob/master/mobilenetv3_small.py) and [large](https://github.com/bisonai/mobilenetv2-tensorflow/blob/master/mobilenetv3_large.py) MobileNetV3 architecture implemented using TensforFlow with `tf.keras` API.

## Requirements
* Python 3.6+
* TensorFlow 1.13+

## Example

### MobileNetV3 Small
```python
from mobilenetv3_small import MobileNetV3 as mobilenetv3_small
model = mobilenetv3_small(
    num_classes=1001,
    width_multiplier=1.0,
)
```

### MobileNetV3 Large
```python
from mobilenetv3_large import MobileNetV3 as mobilenetv3_large
model = mobilenetv3_large(
    num_classes=1001,
    width_multiplier=1.0,
)
```

## License
[Apache License 2.0](https://github.com/bisonai/mobilenetv3-tensorflow/blob/master/LICENSE)
