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

from utils import get_layer


class Identity(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Identity")

    def call(self, input):
        return input


class ReLU6(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="ReLU6")
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
    def __init__(self, name="HardSwish"):
        super().__init__(name=name)
        self.hard_sigmoid = HardSigmoid()

    def call(self, input):
        return input * self.hard_sigmoid(input)


class Squeeze(tf.keras.layers.Layer):
    """Squeeze the second and third dimensions of given tensor.
    (batch, 1, 1, channels) -> (batch, channels)
    """
    def __init__(self):
        super().__init__(name="Squeeze")

    def call(self, input):
        x = tf.keras.backend.squeeze(input, 1)
        x = tf.keras.backend.squeeze(x, 1)
        return x


class GlobalAveragePooling2D(tf.keras.layers.Layer):
    """Return tensor of output shape (batch_size, rows, cols, channels)
    where rows and cols are equal to 1. Output shape of
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

        super().build(input_shape)

    def call(self, input):
        return self.gap(input)


class BatchNormalization(tf.keras.layers.Layer):
    """Searching fo MobileNetV3: All our convolutional layers
    use batch-normalization layers with average decay of 0.99.
    """
    def __init__(
            self,
            momentum: float=0.99,
            name="BatchNormalization",
    ):
        super().__init__(name=name)

        self.bn = tf.keras.layers.BatchNormalization(
            momentum=0.99,
            name="BatchNormalization",
        )

    def call(self, input):
        return self.bn(input)


class ConvNormAct(tf.keras.layers.Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: int=3,
            stride: int=1,
            padding: int=0,
            norm_layer: str=None,
            act_layer: str="relu",
            use_bias: bool=True,
            l2_reg: float=1e-5,
            name: str="ConvNormAct",
    ):
        super().__init__(name=name)

        if padding > 0:
            self.pad = tf.keras.layers.ZeroPadding2D(
                padding=padding,
                name=f"Padding{padding}x{padding}",
            )
        else:
            self.pad = Identity()

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            name=f"Conv{kernel_size}x{kernel_size}",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            use_bias=use_bias,
        )

        _available_normalization = {
            "bn": BatchNormalization(),
            }
        self.norm = get_layer(norm_layer, _available_normalization, Identity())

        _available_activation = {
            "relu": tf.keras.layers.ReLU(name="ReLU"),
            "relu6": ReLU6(),
            "hswish": HardSwish(),
            "hsigmoid": HardSigmoid(),
            "softmax": tf.keras.layers.Softmax(name="Softmax"),
        }
        self.act = get_layer(act_layer, _available_activation, Identity())

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
            act_layer: str,
            l2_reg: float=1e-5,
    ):
        super().__init__(name="Bneck")

        self.out_channels = out_channels
        self.stride = stride
        self.use_se = use_se

        # Expand
        self.expand = ConvNormAct(
            exp_channels,
            kernel_size=1,
            norm_layer="bn",
            act_layer=act_layer,
            use_bias=False,
            l2_reg=l2_reg,
            name="Expand",
        )

        # Depthwise
        dw_padding = (kernel_size - 1) // 2
        self.pad = tf.keras.layers.ZeroPadding2D(
            padding=dw_padding,
            name=f"Depthwise/Padding{dw_padding}x{dw_padding}",
        )
        self.depthwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=stride,
            name=f"Depthwise/DWConv{kernel_size}x{kernel_size}",
            depthwise_regularizer=tf.keras.regularizers.l2(l2_reg),
            use_bias=False,
        )
        self.bn = BatchNormalization(name="Depthwise/BatchNormalization")
        if self.use_se:
            self.se = SEBottleneck(
                l2_reg=l2_reg,
                name="Depthwise/SEBottleneck",
            )

        _available_activation = {
            "relu": tf.keras.layers.ReLU(name="Depthwise/ReLU"),
            "hswish": HardSwish(name="Depthwise/HardSwish"),
        }
        self.act = get_layer(act_layer, _available_activation, Identity())

        # Project
        self.project = ConvNormAct(
            out_channels,
            kernel_size=1,
            norm_layer="bn",
            act_layer=None,
            use_bias=False,
            l2_reg=l2_reg,
            name="Project",
        )

    def build(self, input_shape):
        self.in_channels = int(input_shape[3])
        super().build(input_shape)

    def call(self, input):
        x = self.expand(input)
        x = self.pad(x)
        x = self.depthwise(x)
        x = self.bn(x)
        if self.use_se:
            x = self.se(x)
        x = self.act(x)
        x = self.project(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            return input + x
        else:
            return x


class SEBottleneck(tf.keras.layers.Layer):
    def __init__(
            self,
            reduction: int=4,
            l2_reg: float=0.01,
            name: str="SEBottleneck",
    ):
        super().__init__(name=name)

        self.reduction = reduction
        self.l2_reg = l2_reg

    def build(self, input_shape):
        input_channels = int(input_shape[3])
        self.gap = GlobalAveragePooling2D()
        self.conv1 = ConvNormAct(
            input_channels // self.reduction,
            kernel_size=1,
            norm_layer=None,
            act_layer="relu",
            use_bias=False,
            l2_reg=self.l2_reg,
            name="Squeeze",
        )
        self.conv2 = ConvNormAct(
            input_channels,
            kernel_size=1,
            norm_layer=None,
            act_layer="hsigmoid",
            use_bias=False,
            l2_reg=self.l2_reg,
            name="Excite",
        )

        super().build(input_shape)

    def call(self, input):
        x = self.gap(input)
        x = self.conv1(x)
        x = self.conv2(x)
        return input * x


class LastStage(tf.keras.layers.Layer):
    def __init__(
            self,
            penultimate_channels: int,
            last_channels: int,
            num_classes: int,
            l2_reg: float,
    ):
        super().__init__(name="LastStage")

        self.conv1 = ConvNormAct(
            penultimate_channels,
            kernel_size=1,
            stride=1,
            norm_layer="bn",
            act_layer="hswish",
            use_bias=False,
            l2_reg=l2_reg,
        )
        self.gap = GlobalAveragePooling2D()
        self.conv2 = ConvNormAct(
            last_channels,
            kernel_size=1,
            norm_layer=None,
            act_layer="hswish",
            l2_reg=l2_reg,
        )
        self.dropout = tf.keras.layers.Dropout(
            rate=0.2,
            name="Dropout",
        )
        self.conv3 = ConvNormAct(
            num_classes,
            kernel_size=1,
            norm_layer=None,
            act_layer="softmax",
            l2_reg=l2_reg,
        )
        self.squeeze = Squeeze()

    def call(self, input):
        x = self.conv1(input)
        x = self.gap(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.squeeze(x)
        return x
