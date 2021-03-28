import torch


def check_accuracy(
    loader, model, input_shape=None, toggle_eval=True, print_accuracy=True
):
    """
    Check accuracy of a PyTorch model on a dataloader. It assumes the input
    of the data input shape and will resize if you specify a input_shape.


    Parameters
    ----------
    loader : DataLoader Class
        Loader of the data you want to check the accuracy on
    model : PyTorch Model
        Trained model
    input_shape : list (default None)
        The shape of one example (not including batch), that it should reshape to,
        if left to default argument None it won't reshape.
    toggle_eval : boolean (default True)
        If the model should be toggled to eval mode before evaluation, will return
        to train mode after checking accuracy.
    print_accuracy : boolean (default True)
        If it should also print the accuracy

    Returns
    -------
    float
        Accuracy of the model
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
