import argparse
import dataclasses
from typing import Union

from transformers import HfArgumentParser

from cerebras.pytorch.utils import CSConfig
from model import ChameleonConfig


@dataclasses.dataclass
class RunConfig:
    out_dir: str = "out"
    data_file: str = "train.bin"
    batch_size: int = 1
    sequence_length: int = 32
    num_steps: int = 1000
    checkpoint_steps: int = 100
    learning_rate: float = 1e-4
    warmup_steps: int = 40
    lr_decay: float = 0.999
    weight_decay: float = 0.1
    gradient_clip_val: float = 1.0
    backend: str = "CPU"
    checkpoint_path: str = None
    seed: int = 42
    max_gradient_norm: float = 1.0

    def __post_init__(self):
        assert self.backend in ["CSX", "CPU", "GPU"]
        assert 0 < self.warmup_steps
        assert 0 < self.num_steps


def convert_optional_types(t):
    if t == Union[int, None]:
        return int
    if t == Union[str, None]:
        return str
    if t == Union[float, None]:
        return float
    if t == Union[list, None]:
        return list
    return t


def parse_args():
    config_classes = (ChameleonConfig, RunConfig, CSConfig)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    for config_class in config_classes[:-1]:
        class_name = config_class.__name__
        fields = dataclasses.fields(config_class)
        for f in fields:
            parser.add_argument(
                f"--{f.name}",
                type=convert_optional_types(f.type),
                dest=f"{class_name}.{f.name}",
            )
    args = parser.parse_args()

    hf_parser = HfArgumentParser(config_classes)
    configs = hf_parser.parse_yaml_file(args.config_file)

    new_configs = []
    args = vars(args)
    for config, config_class in zip(configs, config_classes):
        class_name = config_class.__name__
        kws = {}
        for k in args:
            if k.startswith(class_name) and args[k] is not None:
                field = k[len(class_name) + 1 :]
                kws[field] = args[k]
        if kws:
            config = dataclasses.replace(config, **kws)
        new_configs.append(config)

    return new_configs