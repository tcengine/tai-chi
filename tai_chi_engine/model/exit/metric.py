__all__ = ["accuracy", "bi_accuracy"]


def accuracy(y_, y):
    return (y_.argmax(-1) == y).float().mean()


def bi_accuracy(y_, y):
    return ((y_ > .5).float() == y).float().mean()
