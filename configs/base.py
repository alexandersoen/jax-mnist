import dataclasses
from abc import ABC, abstractmethod
from typing import Any


@dataclasses.dataclass(unsafe_hash=True)
class ObjectConfig(ABC):

    @property
    @abstractmethod
    def object_class(self) -> str:
        pass

    def to_params(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_dict(self) -> dict[str, Any]:
        param_dict = self.to_params()
        param_dict["object_class"] = self.object_class
        return param_dict


@dataclasses.dataclass(unsafe_hash=True)
class ModelConfig(ObjectConfig):
    pass


@dataclasses.dataclass(unsafe_hash=True)
class Config:
    dataset: str

    model_config: ModelConfig

    learning_rate: float
    momentum: float

    num_epochs: int
    log_every_steps: int

    batch_size: int
    cache: bool
    shuffle_buffer_size: int

    def to_dict(self) -> dict[str, Any]:
        config_dict = dataclasses.asdict(self)
        config_dict["model_config"] = self.model_config.to_dict()

        return config_dict
