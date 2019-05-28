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

Evaluation script
"""
from argparse import ArgumentParser

import tensorflow as tf

from mobilenetv3_factory import build_mobilenetv3
from cifar10 import cifar10
from mnist import mnist


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


def main(args):
    _available_datasets = {
        "mnist": mnist,
        "cifar10": cifar10,
    }

    if args.dataset not in _available_datasets:
        raise NotImplementedError

    _, _, test_data, num_test_data = _available_datasets.get(args.dataset)(
        args.valid_batch_size,
        args.valid_batch_size,
        args.height,
        args.width,
    )

    model = build_mobilenetv3(
        args.model_type,
        input_shape=(args.height, args.width, args.channels),
        num_classes=args.num_classes,
        width_multiplier=args.width_multiplier,
    )

    _available_optimizers = {
        "rmsprop": tf.train.RMSPropOptimizer,
        "adam": tf.train.AdamOptimizer,
        "sgd": tf.train.GradientDescentOptimizer,
        }

    if args.optimizer not in _available_optimizers:
        raise NotImplementedError

    model.load_weights(args.model_path)

    model.compile(
        optimizer=_available_optimizers.get(args.optimizer)(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.evaluate(
        test_data.make_one_shot_iterator(),
        steps=(num_test_data//args.valid_batch_size)+1,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--width_multiplier", type=float, default=1.0)
    parser.add_argument("--model_type", type=str, default="small", choices=["small", "large"])

    # Input
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=["sgd", "adam", "rmsprop"])

    # Training & validation
    parser.add_argument("--valid_batch_size", type=int, default=256)

    # Others
    parser.add_argument("--model_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
