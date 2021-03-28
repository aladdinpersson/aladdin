import torch


def check_accuracy(
    loader, model, input_shape=None, toggle_eval=True, print_accuracy=True
):
    """
    Check accuracy of model on data from loader
    """
    if toggle_eval:
        model.eval()
    device = next(model.parameters()).device
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            if input_shape:
                x = x.reshape(x.shape[0], *input_shape)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    accuracy = num_correct / num_samples
    if toggle_eval:
        model.train()
    if print_accuracy:
        print(f"Accuracy on training set: {accuracy * 100:.2f}%")
    return accuracy
