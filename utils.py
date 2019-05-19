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

Utility functions
"""
import tensorflow as tf


def _make_divisible(v, divisor, min_value=None):
    """https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


def get_layer(layer_name, layer_dict, default_layer):
    if layer_name is None:
        return default_layer

    if layer_name in layer_dict.keys():
        return layer_dict.get(layer_name)
    else:
        raise NotImplementedError(f"Layer [{layer_name}] is not implemented")


class LayerNamespaceWrapper(tf.keras.layers.Layer):
    """`NameWrapper` defines auxiliary layer that wraps given `layer`
    with given `name`. This is useful for better visualization of network
    in TensorBoard.

    Default behavior of namespaces defined with nested `tf.keras.Sequential`
    layers is to keep only the most high-level `tf.keras.Sequential` name.
    """
    def __init__(
            self,
            layer: tf.keras.layers.Layer,
            name: str,
    ):
        super().__init__(name=name)
        self.wrapped_layer = tf.keras.Sequential(
            [
                layer,
            ],
            name=name,
        )

    def call(self, input):
        return self.wrapped_layer(input)
