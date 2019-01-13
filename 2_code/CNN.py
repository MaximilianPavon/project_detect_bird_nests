from my_functions import split_dataframe, plot_confusion_matrix, plot_history

import pandas as pd
import numpy as np

from sklearn.utils import class_weight
from sklearn import metrics

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
from keras import regularizers, optimizers


if __name__ == '__main__':
    path_to_csv = '../1_data/nests.csv'
    path_to_img = '../1_data/Max_20Flights2017/frames/'
    train_p, val_p = 0.8, 0.1
    index_col = 'files'
    label_col = 'nest'
    img_size = (336, 256)
    color_mode = 'grayscale'
    class_mode = 'categorical'
    batch_size = 32
    n_epochs = 200

    df = pd.read_csv(path_to_csv)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(df[label_col]),
                                                      df[label_col])
    # split data frame into train, validation and test
    df_train, df_val, df_test = split_dataframe(df, train_p, val_p)
    del df

    datagen = ImageDataGenerator()

    train_generator = datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=path_to_img,
        x_col=index_col,
        y_col=label_col,
        has_ext=True,
        target_size=img_size,
        color_mode=color_mode,
        class_mode=class_mode,
        batch_size=batch_size,
        # shuffle=True
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=path_to_img,
        x_col=index_col,
        y_col=label_col,
        has_ext=True,
        target_size=img_size,
        color_mode=color_mode,
        class_mode=class_mode,
        batch_size=batch_size,
        # shuffle=True
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe=df_test,
        directory=path_to_img,
        x_col=index_col,
        y_col=label_col,
        has_ext=True,
        target_size=img_size,
        color_mode=color_mode,
        class_mode=class_mode,
        batch_size=batch_size,
        # shuffle=True
    )

    # build model
    model = Sequential(name='CNN')
    model.add(Conv2D(
        filters=8,
        kernel_size=(5, 5),
        padding='same',
        strides=2,
        activation='relu',
        input_shape=img_size + (1,)
    ))
    model.add(BatchNormalization())

    model.add(Conv2D(
        filters=8,
        kernel_size=(5, 5),
        padding='same',
        strides=2,
        activation='relu',
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=8,
        kernel_size=(5, 5),
        padding='same',
        strides=2,
        activation='relu',
    ))
    model.add(BatchNormalization())
    model.add(Conv2D(
        filters=8,
        kernel_size=(5, 5),
        padding='same',
        strides=2,
        activation='relu',
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.summary()
    plot_model(model, to_file='../3_runs/plots/cnn.png', show_shapes=True)

    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6), loss="categorical_crossentropy", metrics=["accuracy"])

    # fitting the model
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = val_generator.n // val_generator.batch_size

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=val_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  epochs=n_epochs,
                                  verbose=1,
                                  class_weight=class_weights
                                  )

    # plot history
    plot_history(history, n_epochs)

    # evaluate the model
    score = model.evaluate_generator(generator=val_generator)

    # print loss and accuracy
    print('Val loss:', score[0])
    print('Val accuracy:', score[1])

    # Predict the output
    test_generator.reset()
    pred = model.predict_generator(test_generator, verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)

    # Compute confusion matrix
    cnf_matrix = metrics.confusion_matrix(df_test[label_col], predicted_class_indices)
    plot_confusion_matrix(cnf_matrix, classes=[0, 1], title='Confusion matrix, without normalization' + str(n_epochs))

    test_acc = metrics.accuracy_score(df_test[label_col], predicted_class_indices)
    test_precision = metrics.precision_score(df_test[label_col], predicted_class_indices, average='binary')
    test_recall = metrics.recall_score(df_test[label_col], predicted_class_indices, average='binary')
    test_f1 = metrics.f1_score(df_test[label_col], predicted_class_indices, average='binary')

    print('Test accuracy:', test_acc)
    print(metrics.classification_report(df_test[label_col], predicted_class_indices))
