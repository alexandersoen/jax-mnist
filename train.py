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

"""MNIST example.

Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

# See issue #620.
# pytype: disable=wrong-keyword-args

import pathlib
from typing import Any

import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from clu import metric_writers, periodic_actions
from flax import nnx
from flax.training import train_state, checkpoints


import models
from configs.default import TrainConfig
from input_pipeline import create_split, gen_image_processer

NUM_CLASSES = 10


def create_train_state(
    model: nnx.Module, config: TrainConfig
) -> train_state.TrainState:
    """Creates initial `TrainState`."""

    graphdef, params = nnx.split(model, nnx.Param)
    tx = optax.sgd(config.learning_rate, config.momentum)

    if not isinstance(graphdef, nnx.graph.NodeDef):
        raise TypeError("graphdef is not nnx.graph.NodeDef")

    return train_state.TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=tx,
    )


@nnx.jit
def train_step(state: train_state.TrainState, metrics: nnx.MultiMetric, batch):
    inputs = batch["image"]
    labels = batch["label"]

    def loss_fn(params):
        logits, _ = state.apply_fn(params)(inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    metrics.update(loss=loss, logits=logits, labels=labels)
    state = state.apply_gradients(grads=grads)

    return state, metrics


@nnx.jit
def test_step(state: train_state.TrainState, metrics: nnx.MultiMetric, batch):
    inputs = batch["image"]
    labels = batch["label"]

    logits, _ = state.apply_fn(state.params)(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    metrics.update(loss=loss, logits=logits, labels=labels)
    return metrics


def log_summary(summary: dict[str, Any]) -> None:
    # Summarize the scores as info
    logging.info(
        (
            "[train] "
            f"epoch: {summary['epoch']}, "
            f"step: {summary['step']}, "
            f"loss: {summary['train_loss']}, "
            f"accuracy: {summary['train_accuracy']}"
        )
    )

    logging.info(
        (
            "[test] "
            f"epoch: {summary['epoch']}, "
            f"step: {summary['step']}, "
            f"loss: {summary['test_loss']}, "
            f"accuracy: {summary['test_accuracy']}"
        )
    )


def train_and_evaluate(
    config: TrainConfig, workdir: pathlib.Path
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    ###########################################################################

    tf.random.set_seed(0)

    ###########################################################################

    dataset_builder = tfds.builder(config.dataset_config.name)
    dataset_builder.download_and_prepare()

    image_process = gen_image_processer(
        config.dataset_config.input_key, config.dataset_config.target_key
    )
    train_split = create_split(
        dataset_builder,
        ds_part=config.dataset_config.train_partition,
        process=image_process,
        batch_size=config.batch_size,
        cache=config.cache,
        shuffle_buffer_size=config.shuffle_buffer_size,
        shuffle=True,
        repeat=True,
    )
    test_split = create_split(
        dataset_builder,
        ds_part=config.dataset_config.test_partition,
        process=image_process,
        batch_size=config.batch_size,
        cache=config.cache,
        shuffle_buffer_size=None,
        shuffle=False,
        repeat=False,
    )

    ###########################################################################
    # Calculate number of steps for each of these
    train_ds = train_split.ds
    test_ds = test_split.ds

    train_steps_per_epoch = train_split.num_examples_per_epoch // config.batch_size
    num_train_steps = config.num_epochs * train_steps_per_epoch
    num_test_steps = test_split.num_examples_per_epoch // config.batch_size

    ###########################################################################

    # Get corresponding model from config file
    model: nnx.Module = getattr(models, config.model_config.object_class)(
        **config.model_config.to_params(), rngs=nnx.Rngs(0)
    )
    state = create_train_state(model, config=config)

    ###########################################################################

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    ###########################################################################

    # Define writer + similar objects
    writer = metric_writers.create_default_writer(workdir)

    # Record learning settings
    writer.write_hparams(config.to_dict())
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=num_train_steps, writer=writer
    )

    ###########################################################################

    with metric_writers.ensure_flushes(writer):
        for step, batch in zip(range(num_train_steps), train_ds.as_numpy_iterator()):
            is_last_step = step == num_train_steps - 1

            ###########################################################################

            # Train step
            state, metrics = train_step(state, metrics, batch)
            report_progress(step)

            ###########################################################################

            # Check for logging
            if (step % config.log_every_steps == 0) or is_last_step:
                # epoch = step // train_steps_per_epoch
                # summary: dict[str, Any] = {"epoch": epoch, "step": step}

                # Process train stats
                with report_progress.timed("train_metrics"):
                    train_summary: dict[str, Any] = {}

                    for metric, value in metrics.compute().items():
                        train_summary[f"train_{metric}"] = value

                    writer.write_scalars(step, train_summary)
                    metrics.reset()

                # Process test stats
                with report_progress.timed("test_metrics"):
                    test_summary: dict[str, Any] = {}

                    for _, test_batch in zip(
                        range(num_test_steps), test_ds.as_numpy_iterator()
                    ):
                        metrics = test_step(state, metrics, test_batch)

                    for metric, value in metrics.compute().items():
                        test_summary[f"test_{metric}"] = value

                    writer.write_scalars(step, test_summary)
                    metrics.reset()

            # Save a checkpoint on one host after every checkpoint_freq steps.
            save_checkpoint = step % config.checkpoint_every_steps == 0 or is_last_step
            if (config.checkpoint_every_steps > 0) and save_checkpoint:
                logging.info("Saving checkpoint step %d.", step)
                with report_progress.timed("checkpoint"):
                    checkpoints.save_checkpoint_multiprocess(
                        workdir.resolve(),
                        state,
                        step,
                        keep=config.checkpoint_max_keep,
                    )

        ###########################################################################

        return state
