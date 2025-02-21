import torch

from .bignet import BIGNET_DIM
from .stats import load_model


def compare_model_forward(model1: torch.nn.Module, model2: torch.nn.Module, device: str):
    x = torch.randn(1024, BIGNET_DIM).to(device)
    with torch.no_grad():
        max_diff = float(abs(model1(x) - model2(x)).max())
        mean_diff = float(abs(model1(x) - model2(x)).mean())

    return max_diff, mean_diff


def compare_models(model_name1: str, *model_name2: str):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model1 = load_model(model_name1, "bignet.pth").to(device)
    for m2 in model_name2:
        print(f"Comparing {model_name1} and {m2}")
        model2 = load_model(m2, "bignet.pth").to(device)

        max_diff, mean_diff = compare_model_forward(model1, model2, device)
        print(f" - Max difference: {max_diff:.4f}")
        print(f" - Mean difference: {mean_diff:.4f}")


if __name__ == "__main__":
    from fire import Fire

    Fire(compare_models)
