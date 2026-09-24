import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure()
    plt.plot(history.train_loss, label="train loss")
    plt.plot(history.valid_loss, label="valid loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    plt.figure()
    plt.plot(history.train_accuracy, label="train accuracy")
    plt.plot(history.valid_accuracy, label="valid accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.show()

def plot_label_distribution(labels):
    labels = np.asarray(labels)
    values = [np.sum(labels == 0), np.sum(labels == 1), np.sum(labels == -1)]
    x_points = np.array(["label = 0", "label = 1", "label = -1"])

    plt.figure()
    plt.bar(x_points, values, width=0.8)
    plt.xlabel("labels value")
    plt.ylabel("No. of messages")
    plt.title("No. of messages for each label")
    plt.show()

def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
):
    cm = np.asarray(cm)

    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
