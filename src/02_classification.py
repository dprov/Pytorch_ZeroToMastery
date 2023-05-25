import matplotlib.pyplot as plt
import sklearn.datasets
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import utils

N_CLASSES = 4
N_FEATURES = 2
RAND_SEED = 42


def generate_data(
    n_samples=1000,
    n_features=N_FEATURES,
    n_classes=N_CLASSES,
    cluster_std=1.5,
    test_split=0.2,
    device=None,
    seed=None,
):
    X, Y = sklearn.datasets.make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=cluster_std,
        random_state=seed,
    )

    # plt.figure(figsize=(10, 7))
    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.RdYlBu)
    # plt.show()

    X = torch.from_numpy(X).type(torch.float)
    Y = torch.from_numpy(Y).type(torch.LongTensor)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_split, random_state=seed
    )
    return X_train.to(device), X_test.to(device), Y_train.to(device), Y_test.to(device)


class BlobModel(nn.Module):
    def __init__(
        self, in_features=N_FEATURES, hidden_layers=1, hidden_units=8, out_features=N_CLASSES
    ) -> None:
        super().__init__()
        layers = [
            nn.Linear(in_features=in_features, out_features=hidden_units),
            nn.ReLU(),
        ]
        layers.extend(
            [
                nn.Linear(in_features=hidden_units, out_features=hidden_units),
                nn.ReLU(),
            ]
            * hidden_layers
        )
        layers.append(nn.Linear(in_features=hidden_units, out_features=out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(RAND_SEED)
    torch.cuda.manual_seed(RAND_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Gen data
    X_train, X_test, Y_train, Y_test = generate_data(device=device, seed=RAND_SEED)

    # Instantiate model
    model = BlobModel().to(device)
    utils.training.train_model(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        loss_type=nn.CrossEntropyLoss,
        optimizer_type=torch.optim.SGD,
        pred_post_proc=lambda x: torch.softmax(x, dim=1).argmax(dim=1),
        lr=0.01,
        epochs=1000,
        print_each_n_epoch=100,
    )
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Train")
    utils.plot.plot_decision_boundary(model=model, X=X_train, Y=Y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    utils.plot.plot_decision_boundary(model=model, X=X_train, Y=Y_train)
    plt.show()
