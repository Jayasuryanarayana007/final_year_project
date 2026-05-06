import sklearn.metrics as metrics
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)

def precision(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred)

def recall(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred)

def accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred)


def classification_report(y_true, y_pred):
    return metrics.classification_report(y_true, y_pred)


if __name__ == '__main__':
    try:
        path = sys.argv[1]
    except IndexError:
        print("Please provide a path to the csv file.")
        sys.exit(1)

    record = pd.read_csv(path)
    y_true = list(record['labels'])
    y_pred = list(record['pred_labels'])

    print(f'F1: {f1_score(y_true, y_pred)}')
    print(f'Precision: {precision(y_true, y_pred)}')
    print(f'Recall: {recall(y_true, y_pred)}')
    print(f'Accuracy: {accuracy(y_true, y_pred)}')

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    sns.set(font_scale=2.5)
    label_font = {'size': '18'}  # Adjust to fit

    cm = sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True))
    fig = cm.get_figure()

    cm.set_xlabel('Predicted labels', fontdict=label_font)
    cm.set_ylabel('Observed labels', fontdict=label_font)
    cm.tick_params(labelsize=12)
    fig.tight_layout()

    fig.savefig(f"predictions/confusion_matrix_{path.split('/')[-1].split('.')[0]}.png")