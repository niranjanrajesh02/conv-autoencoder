from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model


def CAE(input_shape=(224, 224, 3),  filters=[32, 64, 128, 10], code_dim=10):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    model.add(Conv2D(32, 5, strides=2, padding='same',
              activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(64, 5, strides=2,
              padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(128, 3, strides=2,
              padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=code_dim, name='embedding'))
    model.add(Dense(
        units=128*int(input_shape[0]/8) * int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), 128)))

    model.add(Conv2DTranspose(
        64, 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(
        32, 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(
        input_shape[2], 5, activation='sigmoid', strides=2, padding='same', name='deconv1'))

    model.summary()
    return model
