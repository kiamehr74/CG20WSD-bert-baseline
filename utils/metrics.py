import numpy as np


def accuracy_from_logits(logits, labels):
    np_logits = logits.detach().cpu().numpy()
    np_labels = labels.detach().cpu().numpy()

    total = len(labels)
    correct = np.sum((np.argmax(np_logits, axis=1) == np_labels).astype('int'))
    return float(correct) / total


