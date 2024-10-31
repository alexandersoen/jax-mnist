from abc import ABC, abstractmethod
import dataclasses
from typing import Any


@dataclasses.dataclass(unsafe_hash=True)
class ObjectConfig(ABC):

    @property
    @abstractmethod
    def object_class(self) -> str:
        pass

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(unsafe_hash=True)
class ModelConfig(ObjectConfig):
    pass


@dataclasses.dataclass(unsafe_hash=True)
class Config:
    dataset: str  # = "toy_regression"

    model_config: ModelConfig

    learning_rate: float  # = 0.1
    momentum: float  # = 0.9

    num_epochs: int  # = 10
    log_every_steps: int  # = 100

    train_val_perc: int  # = 80
    batch_size: int  # = 128
    cache: bool  # = True
    shuffle_buffer_size: int  # = 1024

    def to_dict(self) -> dict[str, Any]:
        config_dict = dataclasses.asdict(self)
        config_dict["model_config"] = self.model_config.to_dict()

        return config_dict
