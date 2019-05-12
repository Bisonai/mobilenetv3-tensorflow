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

MobilenetV3 Small
"""
import tensorflow as tf

from layers import ConvBnAct
from layers import HardSwish
from layers import Bneck
from layers import GlobalAveragePooling2D
from layers import SEBottleneck
from utils import _make_divisible
from utils import LayerNamespaceWrapper


class MobileNetV3(tf.keras.Model):
    def __init__(
            self,
            num_classes: int=1001,
            width_multiplier: float=1.0,
            scope: str="MobileNetV3",
            divisible_by: int=8,
    ):
        super().__init__(name=scope)

        # First layer
        self.first_layer = ConvBnAct(16, kernel_size=3, stride=2, padding=1, act_layer=HardSwish)

        # Bottleneck layers
        self.bneck_settings = [
            # k   exp   out  SE      nl     s
            [ 3,  16,   16,  True,   "RE",  2 ],
            [ 3,  72,   24,  False,  "RE",  2 ],
            [ 3,  88,   24,  False,  "RE",  1 ],
            [ 5,  96,   40,  True,   "HS",  2 ],
            [ 5,  240,  40,  True,   "HS",  1 ],
            [ 5,  240,  40,  True,   "HS",  1 ],
            [ 5,  120,  48,  True,   "HS",  1 ],
            [ 5,  144,  48,  True,   "HS",  1 ],
            [ 5,  288,  96,  True,   "HS",  2 ],
            [ 5,  576,  96,  True,   "HS",  1 ],
            [ 5,  576,  96,  True,   "HS",  1 ],
        ]

        self.bneck = tf.keras.Sequential(name="Bneck")
        for idx, (k, exp, out, SE, NL, s) in enumerate(self.bneck_settings):
            out_channels = _make_divisible(out * width_multiplier, divisible_by)
            exp_channels = _make_divisible(exp * width_multiplier, divisible_by)

            self.bneck.add(
                LayerNamespaceWrapper(
                    Bneck(
                        out_channels=out_channels,
                        exp_channels=exp_channels,
                        kernel_size=k,
                        stride=s,
                        use_se=SE,
                        nl=NL,
                    ),
                    namespace=f"Bneck{idx}")
            )

        # Last stage
        penultimate_channels = _make_divisible(576 * width_multiplier, divisible_by)
        last_channels = _make_divisible(1_280 * width_multiplier, divisible_by)

        self.last_stage = tf.keras.Sequential(
            [
                ConvBnAct(penultimate_channels, kernel_size=1, stride=1, act_layer=HardSwish),
                SEBottleneck(),
                GlobalAveragePooling2D(),
                HardSwish(),
                ConvBnAct(last_channels, kernel_size=1, act_layer=HardSwish),
                ConvBnAct(num_classes, kernel_size=1, act_layer=HardSwish),
            ],
            name="LastStage",
        )

    def call(self, input, training=False):
        x = self.first_layer(input)
        x = self.bneck(x)
        x = self.last_stage(x)
        return x
