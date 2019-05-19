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

MobileNetV3 Factory
"""
from typing import Tuple

import tensorflow as tf

from mobilenetv3_large import MobileNetV3 as mobilenetv3_large
from mobilenetv3_small import MobileNetV3 as mobilenetv3_small


_available_models = {
    "small": mobilenetv3_small,
    "large": mobilenetv3_large,
}


def build_mobilenetv3(
        model_type: str,
        input_shape: Tuple[int, int, int]=(224, 224, 3),
        num_classes: int=1001,
        width_multiplier: float=1.0,
        l2_reg: float=1e-5,
):
    assert len(input_shape) == 3, "`input_shape` should be a tuple representing input data shape (height, width, channels)"

    if model_type not in _available_models.keys():
        raise NotImplementedError

    model = _available_models.get(model_type)(
        num_classes=num_classes,
        width_multiplier=width_multiplier,
        l2_reg=l2_reg,
    )

    input_tensor = tf.keras.layers.Input(shape=input_shape)
    output_tensor = model(input_tensor)

    model = tf.keras.Model(
        inputs=[model.input],
        outputs=[model.output],
    )

    return model
