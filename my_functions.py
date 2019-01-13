# import matplotlib
# matplotlib.use("TkAgg")  # fix for macOS
import matplotlib.pyplot as plt
import itertools
import numpy as np


def split_dataframe(df, train_p, val_p, random_state=200):
    '''
    split data frame into train, validation and test
    '''
    df_train = df.sample(frac=train_p, random_state=random_state)
    df = df.drop(df_train.index)
    df_val = df.sample(frac=val_p / (1 - train_p), random_state=random_state)
    df_test = df.drop(df_val.index)

    # make index_col as indeces
    # df_train = df_train.set_index(index_col)
    # df_val = df_val.set_index(index_col)
    # df_test = df_test.set_index(index_col)

    # reset the index to prevent problems of 0 indexing
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    df_test = df_test.reset_index()

    return df_train, df_val, df_test


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
    plt.clf()
    return


def plot_history(history, n_epochs):
    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('acc_' + str(n_epochs) + '.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss_' + str(n_epochs) + '.png', dpi=300, bbox_inches='tight')
    plt.clf()
    return
