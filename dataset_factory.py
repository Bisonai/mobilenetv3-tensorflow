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
import random
from pathlib import Path
from typing import Tuple
from typing import Callable

import tensorflow as tf


def build_dataset(
        data_path: str,
        parse_function: Callable,
        shape: Tuple[int, int, int],
        split_name: str="train",
        batch_size: int=32,
        image_type: str="*.png",
        training: bool=False,
):
    data_split_path = Path(data_path) / split_name
    labels = [label_path.name for label_path in
              filter(lambda p: p.is_dir(), data_split_path.glob("*"))]

    labels_list = []
    image_paths_list = []

    for label_id in labels:
        label_dir = data_split_path / label_id
        images = [str(p) for p in label_dir.glob(image_type)]
        labels_list.extend([int(label_id)] * len(images))
        image_paths_list.extend(images)

    couple = list(zip(image_paths_list, labels_list))
    random.shuffle(couple)

    image_paths_list, labels_list = tuple(zip(*couple))

    filenames = tf.constant(image_paths_list)
    labels = tf.constant(labels_list)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(lambda filename, label: parse_function(filename, label, shape))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset, len(labels_list)
