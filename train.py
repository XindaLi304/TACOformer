# train.py
import torch
from torch import nn
import pytorch_warmup as warmup

def train_one_epoch(model, train_loader, optimizer, lr_scheduler, use_warmup, log_interval=50):
    """
    Keep your original logic: CrossEntropy, per-batch CosineAnnealing step (if provided),
    optional UntunedLinearWarmup dampening context.
    """
    model.train()
    device_list_style = ['gpu' if torch.cuda.is_available() else 'cpu']
    print(device_list_style)
    criterion = nn.CrossEntropyLoss()
    if device_list_style[0] == 'gpu':
        model.cuda(); criterion.cuda()

    warm_ctx = warmup.UntunedLinearWarmup(optimizer) if use_warmup else None

    for batch_idx, (data, target) in enumerate(train_loader):
        if device_list_style[0] == 'gpu':
            data = data.cuda(); target = target.cuda()
            data = data.to(torch.float32); target = target.to(torch.float32)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            if use_warmup:
                with warm_ctx.dampening():
                    lr_scheduler.step()
            else:
                lr_scheduler.step()

        if batch_idx % log_interval == 0:
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).sum()
            print(f"Train [{batch_idx * len(data):>6d}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)] "
                  f"Loss: {loss.item():.6f}, "
                  f"Accuracy: {correct}/{train_loader.batch_size} "
                  f"({100. * correct / train_loader.batch_size:.0f}%)")

def evaluate(model, loader, fold=None, real_test=False, collect_cm=False):
    """
    Keep your evaluation style: accumulate nn.CrossEntropyLoss over dataset,
    print average, accuracy; optionally collect confusion matrix data.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    model.eval()
    device_list_style = ['gpu' if torch.cuda.is_available() else 'cpu']
    print(device_list_style)
    criterion = nn.CrossEntropyLoss()
    if device_list_style[0] == 'gpu':
        criterion.cuda(); model.cuda()

    test_loss = 0.0
    correct = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(torch.float32); target = target.to(torch.float32)
            if device_list_style[0] == 'gpu':
                data = data.cuda(); target = target.cuda()
                data = data.to(torch.float32); target = target.to(torch.float32)

            output = model(data)
            test_loss += criterion(output, target.long())
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

            if collect_cm:
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

    test_loss /= len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)

    if real_test:
        print("running in test dataset with model trained on the selected hyperparameters:")
        print(f"\nIn Test set: Avg. loss: {test_loss:.4f}, "
              f"Accuracy: {correct}/{len(loader.dataset)} ({acc:.0f}%)\n")
    else:
        if fold is None:
            print(f"\nIn Validation set: Avg. loss: {test_loss:.4f}, "
                  f"Accuracy: {correct}/{len(loader.dataset)} ({acc:.0f}%)\n")
        else:
            print(f"\nIn Fold {fold}, Validation set: Avg. loss: {test_loss:.4f}, "
                  f"Accuracy: {correct}/{len(loader.dataset)} ({acc:.0f}%)\n")

    if collect_cm and len(all_preds) > 0:
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        cm = confusion_matrix(all_targets, all_preds)
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.show()

    return float(acc)
