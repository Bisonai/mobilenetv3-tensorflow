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
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tqdm import tqdm
from skimage import io

from dataset_factory import build_dataset


def download_cifar10(data_path: Path):
    print("Downloading CIFAR10 dataset")
    def save_data(data, labels, split_name):
        split_path = data_path / split_name
        split_path.mkdir(parents=True)
        for label_id in tqdm(range(10)):
            label_path = split_path / str(label_id)
            label_path.mkdir()
            for img_id, img in enumerate(data[labels[:, 0] == label_id]):
                io.imsave(Path(label_path / str(img_id)).with_suffix(".png"), img)

    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.cifar10.load_data()

    save_data(eval_data, eval_labels, "test")
    save_data(train_data, train_labels, "train")


def _parse_cifar10(filename, label, shape):
    height, width = shape
    image_string = tf.read_file(filename)
    img = tf.image.decode_png(image_string, channels=3)
    img = tf.image.resize_images(img, [height, width])
    img = ((tf.cast(img, dtype=tf.float32) / tf.constant(255.0)) - tf.constant(0.5)) * tf.constant(2.0)
    return img, tf.one_hot(label, depth=10)


def build_cifar10(
        batch_size: int,
        shape: Tuple[int, int, int],
        training: bool=False,
        data_path: Path=Path("cifar10"),
) -> Tuple[tf.data.Dataset, int]:

    if not data_path.exists():
        download_cifar10(data_path)

    return build_dataset(
        data_path=data_path,
        parse_function=_parse_cifar10,
        split_name="train" if training else "test",
        batch_size=batch_size,
        training=training,
        shape=shape,
    )


def cifar10(
        train_batch_size: int,
        valid_batch_size: int,
        height: int,
        width: int,
):
    train_data, num_train_data = build_cifar10(
        batch_size=train_batch_size,
        shape=(height, width),
        training=True,
    )

    test_data, num_test_data = build_cifar10(
        batch_size=valid_batch_size,
        shape=(height, width),
        training=False,
    )

    return train_data, num_train_data, test_data, num_test_data
