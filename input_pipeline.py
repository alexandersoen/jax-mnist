import enum
import tensorflow as tf
import dataclasses


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


def _process(sample):
    return {
        "image": tf.cast(sample["image"], tf.float32) / 255.0,
        "label": sample["label"],
    }


def create_split(
    dataset_builder,
    ds_part: DatasetPart | str,
    train_val_perc: int,
    batch_size: int,
    cache: bool,
    shuffle_buffer_size: int | None,
):
    """Creates the MNIST train dataset using TensorFlow Datasets.

    Args:
      dataset_builder: TODO
      train_val_perc: TODO
      batch_size: the batch size returned by the data pipeline.
      cache: Whether to cache the dataset.
      shuffle_buffer_size: Size of the shuffle buffer.
      prefetch: Number of items to prefetch in the dataset.
    Returns:
      A `tf.data.Dataset`.
    """
    assert 0 < train_val_perc <= 100

    if type(ds_part) is str:
        ds_part = DatasetPart[ds_part.lower()]

    match ds_part:
        case DatasetPart.train:
            split_str = f"train[:{train_val_perc}%]"
        case DatasetPart.val:
            split_str = f"train[{train_val_perc}%:]"
        case DatasetPart.test:
            split_str = "test"
        case _:
            raise ValueError(f"Unknown dataset part option: {ds_part}")

    num_examples_per_epoch = dataset_builder.info.splits[
        split_str
    ].num_examples

    ds = dataset_builder.as_dataset(split=split_str)

    if cache:
        ds = ds.cache()

    if ds_part == DatasetPart.train:
        ds = ds.repeat()
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.map(_process, tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if ds_part != DatasetPart.train:
        ds = ds.repeat()

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return DatasetSplit(ds, num_examples_per_epoch)
