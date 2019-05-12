# Copyright 2019 Bisonai Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of paper Searching for MobileNetV3, https://arxiv.org/abs/1905.02244

Layers of MobileNetV3
"""
import tensorflow as tf


class Identity(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Identity")

    def call(self, input):
        return input


class ReLU6(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Relu6")
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name="ReLU6")

    def call(self, input):
        return self.relu6(input)


class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="HardSigmoid")
        self.relu6 = ReLU6()

    def call(self, input):
        return self.relu6(input + 3.0) / 6.0


class HardSwish(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="HardSwish")
        self.hard_sigmoid = HardSigmoid()

    def call(self, input):
        return input * self.hard_sigmoid(input)


class GlobalAveragePooling2D(tf.keras.layers.Layer):
    """Return output shape (batch_size, rows, cols, channels).
   `tf.keras.layer.GlobalAveragePooling2D` is (batch_size, channels),
    """
    def __init__(self):
        super().__init__(name="GlobalAveragePooling2D")

    def build(self, input_shape):
        pool_size = tuple(map(int, input_shape[1:3]))
        self.gap = tf.keras.layers.AveragePooling2D(
            pool_size=pool_size,
            name=f"AvgPool{pool_size[0]}x{pool_size[1]}",
        )

    def call(self, input):
        return self.gap(input)


class BatchNormalization(tf.keras.layers.Layer):
    """All our convolutional layers use batch-normalization
    layers with average decay of 0.99.
    """
    def __init__(self):
        super().__init__(name="BatchNormalization")
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=0.99,
            name="BatchNorm",
        )

    def call(self, input):
        return self.bn(input)


class ConvBnAct(tf.keras.layers.Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: int=3,
            stride: int=1,
            padding: int=0,
            act_layer: tf.keras.layers.Layer=tf.keras.layers.ReLU,
    ):
        super().__init__(name="ConvBnAct")

        if padding <= 0:
            self.pad = Identity()
        else:
            self.pad = tf.keras.layers.ZeroPadding2D(
                padding=padding,
                name=f"Padding{padding}x{padding}",
            )
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            name=f"Conv{kernel_size}x{kernel_size}",
        )

        self.norm = BatchNormalization()
        self.act = act_layer()

    def call(self, input):
        x = self.pad(input)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Bneck(tf.keras.layers.Layer):
    def __init__(
            self,
            out_channels: int,
            exp_channels: int,
            kernel_size: int,
            stride: int,
            use_se: bool,
            nl: str,
            act_layer: tf.keras.layers.Layer=ReLU6,
    ):
        super().__init__(name="Bneck")

        self.out_channels = out_channels
        self.exp_channels = exp_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if use_se:
            self.SELayer = SEBottleneck()
        else:
            self.SELayer = Identity()

        if nl.lower() == "re":
            self.nl_layer = ReLU6()
        elif nl.lower() == "hs":
            self.nl_layer = HardSwish()
        else:
            raise NotImplementedError

    def build(self, input_shape):
        self.use_res = self.stride == 1 and int(input_shape[3]) == self.out_channels

        self.pw1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.exp_channels,
                    kernel_size=1,
                    strides=1,
                    name="Conv1x1",
                ),
                BatchNormalization(),
                self.nl_layer,
            ],
            name="PointWise",
        )

        dw_padding = (self.kernel_size - 1) // 2
        self.dw = tf.keras.Sequential(
            [
                tf.keras.layers.ZeroPadding2D(
                    padding=dw_padding,
                    name=f"Padding_{dw_padding}x{dw_padding}",
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=self.kernel_size,
                    strides=self.stride,
                    name=f"DWConv{self.kernel_size}x{self.kernel_size}",
                ),
                BatchNormalization(),
                self.SELayer,
                self.nl_layer,
            ],
            name="DepthWise",
        )

        self.pw2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.out_channels,
                    kernel_size=1,
                    strides=1,
                    name="Conv1x1",
                ),
                BatchNormalization(),
            ],
            name="PointWise",
        )

    def call(self, input):
        x = self.pw1(input)
        x = self.dw(x)
        x = self.pw2(x)

        if self.use_res:
            return input + x
        else:
            return x


class SEBottleneck(tf.keras.layers.Layer):
    def __init__(
            self,
            reduction: int=4,
    ):
        super().__init__(name="SEBottleneck")
        self.reduction = reduction

    def build(self, input_shape):
        input_channels = int(input_shape[3])

        self.se = tf.keras.Sequential(
            [
                GlobalAveragePooling2D(),
                tf.keras.layers.Dense(input_channels // self.reduction),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(input_channels),
                HardSigmoid(),
            ],
            name="SEBottleneck",
        )

    def call(self, input):
        return input * self.se(input)
