import matplotlib.pyplot as plt

def list2str(liste):
    """
    It converts list of sublists into a list
        Inputs:
            liste -> list: input list
        Outputs:
            out -> list: sublists
    """

    out = []
    for each in liste:
        for i in each:
            out.append(i)

    return out

def plot_curve(train_losses, num_epochs):
    """
    It plots the train loss curve. x and y axises represents 
    'epoch number' and 'loss' respectively.
        
        Inputs:
            train_losses -> list: train losses for each epoch
            num_epochs -> int: number of training epochs
    """

    x_axis = range(1, num_epochs)
    y_axis = train_losses

    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis, color="blue", label="Train Loss")
    ax.legend()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("train_loss_curve.png")
