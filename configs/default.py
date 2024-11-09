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
from configs.base import Config, ModelConfig
from models import CNN


@dataclasses.dataclass(unsafe_hash=True)
class CNNModelConfig(ModelConfig):

    @property
    def object_class(self) -> str:
        return CNN.__name__


def get_config() -> Config:
    model_config = CNNModelConfig()

    return Config(
        dataset="mnist",
        model_config=model_config,
        learning_rate=0.1,
        momentum=0.9,
        num_epochs=10,
        log_every_steps=100,
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
