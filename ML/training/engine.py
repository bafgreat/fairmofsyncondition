
from training.metrics import evaluate_model_metrics
import torch

from training.early_stopping import EarlyStopping


def get_target(batch, target_field):
    y = getattr(batch, target_field)
    y = y.float()
    y = y.reshape(batch.num_graphs, -1)
    return y


def train_one_epoch(model, loader, optimizer, loss_fn, device, target_field):
    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(batch)
        y = get_target(batch, target_field)

        loss = loss_fn(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_graphs += batch_size

    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, target_field):
    model.eval()

    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch)
        y = get_target(batch, target_field)

        loss = loss_fn(logits, y)

        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_graphs += batch_size

    return total_loss / total_graphs


def fit(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    loss_fn,
    device,
    target_field,
    epochs=300,
    patience=30,
    log_interval=10,
    scheduler=None,
    metric_fns=["hit@1", "hit@3", "hit@5"],
):
    model = model.to(device)

    early_stopping = EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            target_field=target_field,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            target_field=target_field,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % log_interval == 0:

            to_print = f"Epoch {epoch:03d} | "+f"train loss: {train_loss:.4f} | "
            if metric_fns is not None:
                metrics_train = evaluate_model_metrics(model=model,loader=train_loader,device=device,target_field=target_field)
                for metric_fn in metric_fns:
                    to_print += f"train {metric_fn}: {metrics_train[metric_fn]:.4f} | "
                to_print += f"val loss: {val_loss:.4f} | "
                metrics_val = evaluate_model_metrics(model=model,loader=val_loader,device=device,target_field=target_field)
                for metric_fn in metric_fns:
                    to_print += f"val {metric_fn}: {metrics_val[metric_fn]:.4f} | "

            print(to_print)
        stop = early_stopping.step(val_loss, model)

        if stop:
            print(f"Early stopping at epoch {epoch}")
            break

    early_stopping.restore_best_weights(model)

    test_loss = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        target_field=target_field,
    )
    to_print = ""
    if metric_fns is not None:
        metrics_test = evaluate_model_metrics(model=model,loader=test_loader,device=device,target_field=target_field)
        for metric_fn in metric_fns:
            to_print += f"test {metric_fn}: {metrics_test[metric_fn]:.4f} | "
        print(to_print)
    print(f"Best val loss: {early_stopping.best_loss:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    return model, history, test_loss