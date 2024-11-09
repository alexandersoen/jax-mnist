import dataclasses
from typing import Callable, cast

import tensorflow as tf
from tensorflow_datasets.core import DatasetBuilder

Processer = Callable[[dict], dict]


@dataclasses.dataclass
class DatasetSplit:
    ds: tf.data.Dataset
    num_examples_per_epoch: int


def create_split(
    dataset_builder: DatasetBuilder,
    ds_part: str,
    process: Processer,
    batch_size: int,
    cache: bool,
    shuffle_buffer_size: int | None,
    shuffle: bool = True,
    repeat: bool = True,
) -> DatasetSplit:
    """Creates the MNIST train dataset using TensorFlow Datasets.

    Args:
      dataset_builder: TODO
      process: TODO
      batch_size: the batch size returned by the data pipeline.
      cache: Whether to cache the dataset.
      shuffle_buffer_size: Size of the shuffle buffer.
      prefetch: Number of items to prefetch in the dataset.
    Returns:
      A `tf.data.Dataset`.
    """
    num_examples_per_epoch = dataset_builder.info.splits[ds_part].num_examples

    ds = dataset_builder.as_dataset(split=ds_part)
    ds = cast(tf.data.Dataset, ds)
    ds = ds.map(process, tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()
        ds = ds.take(num_examples_per_epoch)

    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.batch(batch_size, drop_remainder=True)

    if repeat:
        ds = ds.repeat()

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return DatasetSplit(ds, num_examples_per_epoch)


def gen_image_processer(input_str: str, target_str: str) -> Processer:

    def process(sample: dict) -> dict:
        return {
            "image": tf.divide(tf.cast(sample[input_str], tf.float32), 255.0),
            "label": sample[target_str],
        }

    return process
