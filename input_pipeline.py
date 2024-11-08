import dataclasses
import enum
from typing import Callable, cast

import tensorflow as tf

from tensorflow_datasets.core import DatasetBuilder


class DatasetPart(enum.Enum):
    """
    Simple Enum class to specify dataset portion.
    """

    train = enum.auto()
    val = enum.auto()
    test = enum.auto()


@dataclasses.dataclass
class DatasetSplit:
    ds: tf.data.Dataset
    num_examples_per_epoch: int


def image_process(sample):
    return {
        "image": tf.divide(tf.cast(sample["image"], tf.float32), 255.0),
        "label": sample["label"],
    }


def image_splitter(part: DatasetPart, train_val_perc: int) -> str:
    match part:
        case DatasetPart.train:
            split_str = f"train[:{train_val_perc}%]"
        case DatasetPart.val:
            split_str = f"train[{train_val_perc}%:]"
        case DatasetPart.test:
            split_str = "test"

    return split_str


def create_split(
    dataset_builder: DatasetBuilder,
    process: Callable[[dict], dict],
    splitter: Callable[[DatasetPart, int], str],
    ds_part: DatasetPart | str,
    train_val_perc: int,
    batch_size: int,
    cache: bool,
    shuffle_buffer_size: int | None,
) -> DatasetSplit:
    """Creates the MNIST train dataset using TensorFlow Datasets.

    Args:
      dataset_builder: TODO
      process: TODO
      splitter: TODO
      train_val_perc: TODO
      batch_size: the batch size returned by the data pipeline.
      cache: Whether to cache the dataset.
      shuffle_buffer_size: Size of the shuffle buffer.
      prefetch: Number of items to prefetch in the dataset.
    Returns:
      A `tf.data.Dataset`.
    """
    assert 0 < train_val_perc <= 100

    if isinstance(ds_part, str):
        ds_part = DatasetPart[ds_part.lower()]

    split_str = splitter(ds_part, train_val_perc)
    num_examples_per_epoch = dataset_builder.info.splits[split_str].num_examples

    ds = dataset_builder.as_dataset(split=split_str)
    ds = cast(tf.data.Dataset, ds)
    ds = ds.map(process, tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()
        ds = ds.take(num_examples_per_epoch)

    if ds_part == DatasetPart.train:
        ds = ds.repeat()
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.batch(batch_size, drop_remainder=True)

    if ds_part != DatasetPart.train:
        ds = ds.repeat()

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return DatasetSplit(ds, num_examples_per_epoch)
