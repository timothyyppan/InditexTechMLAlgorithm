import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import unpickle as up
import extract_gz as eg

def load_cifar10_batch(file):
    data_dict = up.unpickle(file)
    X = data_dict[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    Y = data_dict[b'labels']
    return X, np.array(Y)

def train_model():
    gz_file = 'cifar-10-python.tar.gz' 
    eg.extract_gz(gz_file)

    extracted_dir = 'cifar-10-batches-py'
    X_train, y_train = [], []

    for i in range(1, 6):
        data_path = f'{extracted_dir}/data_batch_{i}'
        X, Y = load_cifar10_batch(data_path)
        X_train.append(X)
        y_train.append(Y)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = load_cifar10_batch(f'{extracted_dir}/test_batch')

    X_train, X_test = X_train.astype('float32') / 255, X_test.astype('float32') / 255
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))
    return model
