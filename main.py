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

"""Main file for running the MNIST example.

This file is intentionally kept short. The majority of logic is in libraries
than can be easily tested and imported in Colab.
"""

import pathlib
import typing

import jax
import tensorflow as tf
from absl import app, flags, logging
from clu import platform
from ml_collections import config_flags

import train
from configs import TrainConfig

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def gen_subdir(workdir: pathlib.Path, config: TrainConfig) -> pathlib.Path:
    return pathlib.Path(workdir, config.dataset_config.name)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Create work dir
    workdir_path = gen_subdir(FLAGS.workdir, FLAGS.config)
    workdir_path.mkdir(parents=True, exist_ok=True)

    # Log to file
    _handler = logging.get_absl_handler()
    _handler = typing.cast(logging.ABSLHandler, _handler)
    _handler.use_absl_log_file(log_dir=str(workdir_path))

    logging.set_stderrthreshold(logging.INFO)

    print("Logging to: %s" % logging.get_log_file_name())

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, workdir_path, "workdir"
    )

    config: TrainConfig = FLAGS.config
    train.train_and_evaluate(config, workdir_path)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
