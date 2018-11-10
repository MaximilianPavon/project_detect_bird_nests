import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator


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
    train_p, val_p = 0.8, 0.1
    index_col = 'files'
    label_col = 'nest'
    batch_size = 32

    df = pd.read_csv(path_to_csv)
    # split data frame into train, validation and test
    df_train, df_val, df_test = split_dataframe(df, train_p, val_p)
    del df

    datagen = ImageDataGenerator()

    train_generator = datagen.flow_from_dataframe(
        dataframe=df_train,
        directory='data/Max_20Flights2017/frames/',
        x_col=index_col,
        y_col=label_col,
        has_ext=True,
        target_size=(336, 256),
        color_mode='grayscale',
        class_mode='other',
        batch_size=batch_size,
        # shuffle=True
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=df_val,
        directory='data/Max_20Flights2017/frames/',
        x_col=index_col,
        y_col=label_col,
        has_ext=True,
        target_size=(336, 256),
        color_mode='grayscale',
        class_mode='other',
        batch_size=batch_size,
        # shuffle=True
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe=df_test,
        directory='data/Max_20Flights2017/frames/',
        x_col=index_col,
        y_col=label_col,
        has_ext=True,
        target_size=(336, 256),
        color_mode='grayscale',
        class_mode='other',
        batch_size=batch_size,
        # shuffle=True
    )

    i = 0
    for x, y in train_generator:
        print('x ', x.shape, 'y ', y.shape)
        if i > 10:
            break
        i += 1


    print()


