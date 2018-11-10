import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers


def split_dataframe(df, train_p, val_p,  random_state=200):
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


if __name__ == '__main__':
    path_to_csv = 'nests.csv'
    path_to_img = 'data/Max_20Flights2017/frames/'
    train_p, val_p = 0.8, 0.1
    index_col = 'files'
    label_col = 'nest'
    img_size = (336, 256)
    color_mode = 'grayscale'
    class_mode = 'categorical'
    batch_size = 32

    df = pd.read_csv(path_to_csv)
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
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=img_size + (1,), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.summary()

    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6), loss="categorical_crossentropy", metrics=["accuracy"])

    # fitting the model
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = val_generator.n // val_generator.batch_size

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=val_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=10,
                        verbose=1
                        )

    # evaluate the model
    score = model.evaluate_generator(generator=val_generator)

    # print loss and accuracy
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Predict the output
    test_generator.reset()
    pred = model.predict_generator(test_generator, verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)

