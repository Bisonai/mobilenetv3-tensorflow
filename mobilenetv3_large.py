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

MobileNetV3 Large
"""
import tensorflow as tf

from layers import ConvNormAct
from layers import Bneck
from layers import LastStage
from utils import _make_divisible
from utils import LayerNamespaceWrapper


class MobileNetV3(tf.keras.Model):
    def __init__(
            self,
            num_classes: int=1001,
            width_multiplier: float=1.0,
            name: str="MobileNetV3_Large",
            divisible_by: int=8,
            l2_reg: float=1e-5,
    ):
        super().__init__(name=name)

        # First layer
        self.first_layer = ConvNormAct(
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer="bn",
            act_layer="hswish",
            use_bias=False,
            l2_reg=l2_reg,
            name="FirstLayer",
        )

        # Bottleneck layers
        self.bneck_settings = [
            # k   exp   out   SE      NL         s
            [ 3,  16,   16,   False,  "relu",    1 ],
            [ 3,  64,   24,   False,  "relu",    2 ],
            [ 3,  72,   24,   False,  "relu",    1 ],
            [ 5,  72,   40,   True,   "relu",    2 ],
            [ 5,  120,  40,   True,   "relu",    1 ],
            [ 5,  120,  40,   True,   "relu",    1 ],
            [ 3,  240,  80,   False,  "hswish",  2 ],
            [ 3,  200,  80,   False,  "hswish",  1 ],
            [ 3,  184,  80,   False,  "hswish",  1 ],
            [ 3,  184,  80,   False,  "hswish",  1 ],
            [ 3,  480,  112,  True,   "hswish",  1 ],
            [ 3,  672,  112,  True,   "hswish",  1 ],
            [ 5,  672,  160,  True,   "hswish",  2 ],
            [ 5,  960,  160,  True,   "hswish",  1 ],
            [ 5,  960,  160,  True,   "hswish",  1 ],
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
                        act_layer=NL,
                    ),
                    name=f"Bneck{idx}")
            )

        # Last stage
        penultimate_channels = _make_divisible(960 * width_multiplier, divisible_by)
        last_channels = _make_divisible(1_280 * width_multiplier, divisible_by)

        self.last_stage = LastStage(
            penultimate_channels,
            last_channels,
            num_classes,
            l2_reg=l2_reg,
        )

    def call(self, input):
        x = self.first_layer(input)
        x = self.bneck(x)
        x = self.last_stage(x)
        return x
