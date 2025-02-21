import torch

from .bignet import BIGNET_DIM
from .stats import load_model


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
                best_accuracy = accuracy.item()

    return best_accuracy


def load_model_and_fit_binary_classifier(model_name: str, steps: int = 50):
    model = load_model(model_name, "bignet.pth")
    return fit_binary_classifier(model, steps)


if __name__ == "__main__":
    from fire import Fire

    Fire(load_model_and_fit_binary_classifier)
