# Copyright 2024 The Flax Authors.
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

"""Default Hyperparameter configuration."""

import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class Config:
    dataset: str  # = "toy_regression"

    learning_rate: float  # = 0.1
    momentum: float  # = 0.9

    num_epochs: int  # = 10
    log_every_steps: int  # = 100

    train_val_perc: int  # = 80
    batch_size: int  # = 128
    cache: bool  # = True
    shuffle_buffer_size: int  # = 1024


def get_config() -> Config:
    return Config(
        dataset="toy_regression",
        learning_rate=0.1,
        momentum=0.9,
        num_epochs=10,
        log_every_steps=100,
        train_val_perc=80,
        batch_size=128,
        cache=True,
        shuffle_buffer_size=1024,
    )


def metrics():
    return [
        "train_loss",
        "eval_loss",
        "train_accuracy",
        "eval_accuracy",
        "steps_per_second",
    ]
