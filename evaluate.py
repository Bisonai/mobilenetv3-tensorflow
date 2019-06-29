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
from datasets import build_dataset


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

_available_datasets = [
    "mnist",
    "cifar10",
    ]

_available_optimizers = {
    "rmsprop": tf.train.RMSPropOptimizer,
    "adam": tf.train.AdamOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    }

def main(args):
    if args.dataset not in _available_datasets:
        raise NotImplementedError

    dataset = build_dataset(
        name=args.dataset,
        shape=[args.height, args.width],
        )

    model = build_mobilenetv3(
        args.model_type,
        input_shape=(args.height, args.width, dataset["channels"]),
        num_classes=dataset["num_classes"],
        width_multiplier=args.width_multiplier,
    )


    if args.optimizer not in _available_optimizers:
        raise NotImplementedError

    model.load_weights(args.model_path)

    model.compile(
        optimizer=_available_optimizers.get(args.optimizer)(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.evaluate(
        dataset["test"].make_one_shot_iterator(),
        steps=(dataset["num_test"]//args.valid_batch_size)+1,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model
    parser.add_argument("--width_multiplier", type=float, default=1.0)
    parser.add_argument("--model_type", type=str, default="small", choices=["small", "large"])

    # Input
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="mnist", choices=_available_datasets)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=_available_optimizers.keys())

    # Training & validation
    parser.add_argument("--valid_batch_size", type=int, default=256)

    # Others
    parser.add_argument("--model_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
