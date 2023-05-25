from typing import Callable

import torch


def _print_progress(
    epoch: int,
    model: torch.nn.Module,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    loss_fun: torch.nn.Module,
    train_loss: float,
    pred_post_proc: Callable[[torch.Tensor], torch.Tensor] = None,
):
    model.eval()
    with torch.inference_mode():
        Y_pred = model(X_test)
        # if pred_post_proc:
        #     test_loss = loss_fun(pred_post_proc(Y_pred), Y_test)
        # else:
        test_loss = loss_fun(Y_pred, Y_test)
        print(f"Epoch: {epoch} | Loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")


def train_model(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    loss_type: torch.nn.Module,
    optimizer_type: torch.nn.Module,
    pred_post_proc: Callable[[torch.Tensor], torch.Tensor] = None,
    lr=0.01,
    epochs=100,
    print_each_n_epoch=None,
) -> None:
    # Sort out later
    loss_fun = loss_type()
    optimizer = optimizer_type(params=model.parameters(), lr=lr)

    print(f"Training for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        Y_pred = model(X_train)

        # if pred_post_proc:
        #     loss = loss_fun(pred_post_proc(Y_pred), Y_train)
        # else:
        loss = loss_fun(Y_pred, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (
            epoch == 0
            or epoch == epochs - 1
            or (print_each_n_epoch and (epoch % print_each_n_epoch == 0))
        ):
            _print_progress(
                epoch=epoch,
                model=model,
                X_test=X_test,
                Y_test=Y_test,
                loss_fun=loss_fun,
                train_loss=loss,
                pred_post_proc=pred_post_proc,
            )
    print("Training done")
