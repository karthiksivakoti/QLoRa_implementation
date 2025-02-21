from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from . import bignet, half_precision, lora, low_precision, lower_precision, qlora

ALL_MODELS = {
    "bignet": bignet,
    "lora": lora,
    "low_precision": low_precision,
    "lower_precision": lower_precision,
    "half_precision": half_precision,
    "qlora": qlora,
}


def load_model(model_name: str, path: Path | None) -> torch.nn.Module:
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unknown model {model_name}")
    return ALL_MODELS[model_name].load(path)


@dataclass
class MemoryProfile:
    total: int = 0

    def __int__(self) -> int:
        return self.total

    def __str__(self) -> str:
        return f"{self.total / 1024.0 / 1024.0} MB"


@contextmanager
def memory_profile(device):
    if device == "cuda":
        mem = MemoryProfile()
        _mem_init = torch.cuda.memory_allocated()
        yield mem
        _mem_end = torch.cuda.memory_allocated()
        mem.total = _mem_end - _mem_init
    elif device == "mps":
        mem = MemoryProfile()
        _mem_init = torch.mps.current_allocated_memory()
        yield mem
        _mem_end = torch.mps.current_allocated_memory()
        mem.total = _mem_end - _mem_init
    elif device == "cpu":
        from torch.profiler import profile

        mem = MemoryProfile()
        with profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
            yield mem
        mem.total = prof.events().total_average().self_cpu_memory_usage
    else:
        raise ValueError(f"Unknown device {device}")


def num_parameters(model: torch.nn.Module) -> int:
    """
    Number of parameters and buffers in a model.
    """
    from itertools import chain

    return sum(p.numel() for p in chain(model.buffers(), model.parameters()))


def mem_parameters(model: torch.nn.Module) -> int:
    """
    Memory used for parameters and buffers in a model in bytes.
    """
    from itertools import chain

    return sum(p.numel() * p.element_size() for p in chain(model.buffers(), model.parameters()))


@dataclass
class ModelStats:
    num_parameters: int
    "Number of parameters in the model"

    trainable_parameters: int
    "Number of trainable parameters in the model"

    theoretical_memory: float
    "Memory usage of the model in MB (theoretical)"

    actual_memory: float
    "Memory usage of the forward pass in MB"

    forward_memory: float
    "Memory usage of the forward pass in MB"

    backward_memory: float
    "Memory usage of the backward pass in MB"

    @classmethod
    def from_model(cls, m: torch.nn.Module):
        """
        Collect statistics about a model and its memory usage, supporting CPU, CUDA, and MPS.
        """
        original_device = next(m.parameters()).device
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        if device == "mps":
            torch.mps.empty_cache()

        m.to("cpu")
        with memory_profile(device) as mem_model:
            if device == "cpu":
                m_copy = deepcopy(m)
            else:
                m_copy = m.to(device)
        del m_copy

        x = torch.randn(1024, bignet.BIGNET_DIM).to(device)

        with memory_profile(device) as mem_forward:
            with torch.no_grad():
                m(x)

        with memory_profile(device) as mem_backward:
            m(x).mean().backward()

        if device == "mps":
            torch.mps.empty_cache()

        m.to(original_device)
        return cls(
            num_parameters=num_parameters(m),
            trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
            theoretical_memory=mem_parameters(m) / 2**20,
            actual_memory=int(mem_model) / 2**20,
            forward_memory=int(mem_forward) / 2**20,
            backward_memory=int(mem_backward) / 2**20,
        )


def model_info(model_name1: str, *model_name2: str):
    """
    Return the number of parameters and memory usage of a model.
    """
    model_names = [model_name1, *model_name2]
    stats = {}
    for model_name in model_names:
        model = load_model(model_name, None)
        stats[model_name] = ModelStats.from_model(model)

    print("                    ", " ".join([f"{model_name:^14s}" for model_name in stats.keys()]))
    print("Trainable params    ", " ".join([f"  {m.trainable_parameters / 1000000:8.2f} M  " for m in stats.values()]))
    print(
        "Non-trainable params",
        " ".join([f"  {(m.num_parameters - m.trainable_parameters) / 1000000:8.2f} M  " for m in stats.values()]),
    )
    print("Total params        ", " ".join([f"  {m.num_parameters / 1000000:8.2f} M  " for m in stats.values()]))
    print("Theoretical memory  ", " ".join([f"  {m.theoretical_memory:8.2f} MB " for m in stats.values()]))
    print("Actual memory       ", " ".join([f"  {m.actual_memory:8.2f} MB " for m in stats.values()]))
    print("Forward memory      ", " ".join([f"  {m.forward_memory:8.2f} MB " for m in stats.values()]))
    print("Backward memory     ", " ".join([f"  {m.backward_memory:8.2f} MB " for m in stats.values()]))


if __name__ == "__main__":
    from fire import Fire

    Fire(model_info)
