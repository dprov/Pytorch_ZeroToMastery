from typing import Tuple

import torch
import torch.nn as nn

import utils


class LinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.linear_layer(x)


def split_test_train(X: torch.Tensor, test_ratio=0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    ind = int(test_ratio * len(X))
    return X[:ind], X[ind:]


def create_linear_data(
    start=0, end=1, step=0.01, slope=0.7, intercept=0.5, test_train_split=0.8, device=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    Y = slope * X + intercept

    X_train, X_test = split_test_train(X)
    Y_train, Y_test = split_test_train(Y)
    return X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X_train, Y_train, X_test, Y_test = create_linear_data(device=device)

    model = LinearRegression().to(device)
    # Equivalent
    # model = nn.Linear(in_features=1, out_features=1).to(device)

    utils.training.train_model(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        loss_type=torch.nn.L1Loss,
        optimizer_type=torch.optim.SGD,
        epochs=500,
        print_each_n_epoch=100,
        lr=0.01,
    )

    model.eval()
    with torch.inference_mode():
        Y_pred = model(X_test)

    utils.plot.plot_predictions(
        train_data=X_train,
        train_labels=Y_train,
        test_data=X_test,
        test_labels=Y_test,
        predictions=Y_pred,
    )
