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
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Tuple

def build_dataset(
    shape: Tuple[int, int],
    num_classes: int,
    name: str="mnist",
    train_batch_size: int=32,
    valid_batch_size: int=32
    ):
  
    builder = tfds.builder(name)
    num_train = builder.info.splits['train'].num_examples
    num_test = builder.info.splits['test'].num_examples

    ds_train, ds_test = tfds.load(name=name, split=["train", "test"])

    ds_train = ds_train.shuffle(1024).repeat()
    ds_train = ds_train.map(lambda data: _parse_function(data,shape,num_classes))
    ds_train = ds_train.batch(train_batch_size)

    ds_test = ds_test.shuffle(1024).repeat()
    ds_test = ds_test.map(lambda data: _parse_function(data,shape,num_classes))
    ds_test = ds_test.batch(valid_batch_size)

    return ds_train, num_train, ds_test, num_test

def _parse_function(data,shape,num_classes):
    height, width = shape
    image = data["image"]
    label = data["label"]

    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize_images(image, (height,width))
    image = tf.reshape(image, (height,width, 1))
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    label = tf.one_hot(label, depth=num_classes)

    return image, label