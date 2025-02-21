from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .grader import Case, Grader

BIGNET_PTH = Path(__file__).parent.parent / "bignet.pth"
BIGNET_DIM = 1024


def fit_binary_classifier(model: torch.nn.Module, steps: int = 5000):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    # Add binary classifier layer that is not trainable
    classifier = torch.nn.Linear(BIGNET_DIM, 1, bias=False).to(device)
    classifier.requires_grad = False

    # Generate random data
    x = torch.randn(1000, BIGNET_DIM).to(device)
    y = torch.cat([torch.zeros(500), torch.ones(500)]).to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_accuracy = 0

    # Training loop
    model.eval()
    for epoch in range(steps):
        optimizer.zero_grad()

        features = model(x)
        logits = classifier(features)

        loss = criterion(logits.squeeze(), y)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = ((logits.squeeze() > 0) == y).float().mean()
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy

    return best_accuracy


def compare_model_forward(model1: torch.nn.Module, model2: torch.nn.Module, device: str):
    x = torch.randn(1024, BIGNET_DIM).to(device)
    with torch.no_grad():
        max_diff = float(abs(model1(x) - model2(x)).max())
        mean_diff = float(abs(model1(x) - model2(x)).mean())

    return max_diff, mean_diff


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
    def from_model(cls, m: torch.nn.Module, device: str | None = None):
        """
        Collect statistics about a model and its memory usage, supporting CPU, CUDA, and MPS.
        """
        original_device = next(m.parameters()).device

        if device is None:
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

        x = torch.randn(1024, BIGNET_DIM).to(device)

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


class LoraGrader(Grader):
    """LoRA"""

    KIND = "lora"
    TRAIN_STEPS = 20
    ACC_RANGE = 0.50, 0.8
    MEAN_DIFF_BOUND = 5e-4
    MAX_DIFF_BOUND = 5e-3

    TRAINABLE_PARAMS_BOUND = 2 * 1000000  # 2M
    TOTAL_PARAMS_BOUND = 25 * 1000000  # 25M
    TOTAL_MEMORY_BOUND = 50  # MB
    BACKWARD_MEMORY_BOUND = 10  # MB

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def accuracy(
        self, model: torch.nn.Module, min_accuracy: float = 0.5, max_accuracy: float = 1.0
    ) -> tuple[float, str]:
        """
        Returns the accuracy of the model normalized to [0, 1]

        If the accuracy is greater than max_accuracy, you get 1.0 (full score)
        Similarly, if the model's accuracy less than min_accuracy, you get 0.0 (no points)
        """
        model.to(self.device)
        accuracy = fit_binary_classifier(model, self.TRAIN_STEPS)
        accuracy_normalized = (accuracy - min_accuracy) / (max_accuracy - min_accuracy)
        accuracy_normalized = accuracy_normalized.item()
        return np.clip(accuracy_normalized, 0.0, 1.0)

    def load_model(self, model_name: str) -> torch.nn.Module:
        """
        Load a model from the checkpoint
        """
        all_models = {
            "bignet": self.module.bignet,
            "lora": self.module.lora,
            "low_precision": self.module.low_precision,
            "half_precision": self.module.half_precision,
            "qlora": self.module.qlora,
        }
        if model_name not in all_models:
            raise ValueError(f"Unknown model {model_name}")
        return all_models[model_name].load(BIGNET_PTH)

    @Case(score=10, timeout=5000)
    def test_forward_diff(self):
        """Forward accuracy"""
        bigmodel = self.load_model("bignet").to(self.device)
        low_res_model = self.load_model(self.KIND).to(self.device)
        max_diff, mean_diff = compare_model_forward(bigmodel, low_res_model, self.device)
        assert mean_diff < self.MEAN_DIFF_BOUND, f"Mean difference is too high: {mean_diff:.4f}"
        assert max_diff < self.MAX_DIFF_BOUND, f"Max difference is too high: {max_diff:.4f}"

    @Case(score=10, timeout=5000)
    def test_forward_stats(self):
        """Memory and parameters"""
        model = self.load_model(self.KIND)
        stats = ModelStats.from_model(model)

        assert (
            stats.trainable_parameters < self.TRAINABLE_PARAMS_BOUND
        ), f"Trainable parameters are too high: {stats.trainable_parameters:.4f}"
        assert (
            stats.num_parameters < self.TOTAL_PARAMS_BOUND
        ), f"Total parameters are too high: {stats.num_parameters:.4f}"
        assert stats.actual_memory < self.TOTAL_MEMORY_BOUND, f"Actual memory is too high: {stats.actual_memory:.4f}"
        # assert stats.backward_memory < self.BACKWARD_MEMORY_BOUND, f"Backward memory too high: {stats.backward_memory:.4f}"  # noqa: E501

    @Case(score=10, timeout=50000)
    def test_accuracy(self):
        """Backward accuracy"""
        model = self.load_model(self.KIND)
        return self.accuracy(model, *self.ACC_RANGE)


class QLORAGrader(LoraGrader):
    """QLORA"""

    KIND = "qlora"
    MEAN_DIFF_BOUND = 1e-1
    MAX_DIFF_BOUND = 5e-1

    TRAINABLE_PARAMS_BOUND = 1.5 * 1000000  # 1.5M
    TOTAL_PARAMS_BOUND = 15 * 1000000  # 15M
    TOTAL_MEMORY_BOUND = 20  # MB
    BACKWARD_MEMORY_BOUND = 15  # MB


class LowPrecisionGrader(LoraGrader):
    """Low Precision"""

    KIND = "low_precision"

    MEAN_DIFF_BOUND = 1e-1
    MAX_DIFF_BOUND = 5e-1

    TRAINABLE_PARAMS_BOUND = 0.1 * 1000000  # 0.1M
    TOTAL_PARAMS_BOUND = 15 * 1000000  # 15M
    TOTAL_MEMORY_BOUND = 15  # MB
    BACKWARD_MEMORY_BOUND = 10  # MB

    def test_accuracy(self):
        # skip training low-precision model
        pass


class HalfPrecisionGrader(QLORAGrader):
    """Half Precision"""

    ACC_RANGE = 0.90, 1.0
    KIND = "half_precision"
    MEAN_DIFF_BOUND = 1e-3
    MAX_DIFF_BOUND = 1e-2

    TRAINABLE_PARAMS_BOUND = 20 * 1000000  # 20M
    TOTAL_PARAMS_BOUND = 20 * 1000000  # 20M
    TOTAL_MEMORY_BOUND = 80  # MB
    BACKWARD_MEMORY_BOUND = 80  # MB

    def test_accuracy(self):
        # skip training half-precision model
        pass


class ExtraCreditGrader(Grader):
    """Lower Precision"""

    MEAN_DIFF_BOUND = 1e-1
    MAX_DIFF_BOUND = 5e-1

    TOTAL_MEMORY_BOUND = 9  # MB

    # mps is not supported for this extra credit
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_name: str) -> torch.nn.Module:
        """
        Load a model from the checkpoint
        """
        all_models = {
            "bignet": self.module.bignet,
            "lower_precision": self.module.lower_precision,
        }
        if model_name not in all_models:
            raise ValueError(f"Unknown model {model_name}")
        return all_models[model_name].load(BIGNET_PTH)

    @Case(score=5, timeout=5000, extra_credit=True)
    def test_forward_diff(self):
        """Extra Credit"""
        lower_model = self.load_model("lower_precision").to(self.device)
        stats = ModelStats.from_model(lower_model, self.device)

        bigmodel = self.load_model("bignet").to(self.device)
        max_diff, mean_diff = compare_model_forward(bigmodel, lower_model, self.device)
        assert mean_diff < self.MEAN_DIFF_BOUND, f"Mean difference is too high: {mean_diff:.4f}"
        assert max_diff < self.MAX_DIFF_BOUND, f"Max difference is too high: {max_diff:.4f}"

        assert stats.actual_memory < self.TOTAL_MEMORY_BOUND, f"Actual memory is too high: {stats.actual_memory:.4f}"
        assert (
            stats.theoretical_memory < self.TOTAL_MEMORY_BOUND
        ), f"Theoretical memory is too high: {stats.theoretical_memory:.4f}"
