import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

def classification_report_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
    }
