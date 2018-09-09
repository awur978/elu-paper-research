import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.layers import ZeroPadding2D, Cropping2D, Conv2D, MaxPooling2D, Dropout, Activation, Flatten
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import normalize, to_categorical


def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_length):
    '''
    Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator
    '''
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)


def get_datagen(x_train):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0.0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
        brightness_range=None,
        shear_range=0.0,  # set range for random shear
        zoom_range=0.0,  # set range for random zoom
        channel_shift_range=0.0,  # set range for random channel shifts
        fill_mode='nearest',  # set mode for filling points outside the input boundaries
        cval=0.0,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rescale=None,  # set rescaling factor (applied before any other transformation)
        preprocessing_function=None,  # set function that will be applied on each input
        data_format=None,  # image data format, either "channels_first" or "channels_last"
        validation_split=0.0  # fraction of images reserved for validation (strictly between 0 and 1)
    )
    datagen.fit(x_train)
    return datagen


def get_lr_schedule():
    def lr_schedule(epoch, lr):
        if epoch < 35:
            return 0.01
        elif epoch < 85:
            return 0.005
        elif epoch < 135:
            return 0.0005
        else:
            return 0.00005

    return LearningRateScheduler(lr_schedule)


def get_tensorboard(name='cifar100-7blocks-elu-softmax-{}'.format(datetime.datetime.now())):
    tensorboard = TensorBoard(log_dir='./logs/{}'.format(name),
                              # histogram_freq=2,
                              # write_graph=True,
                              # write_grads=True,
                              # write_images=True,
                              )
    return tensorboard


def my_activation(x):
    return tf.where(x > 0.0,
                    tf.log(tf.maximum(x, 0.0) + 1.0),
                    -tf.log(-tf.minimum(x, 0.0) + 1.0))


def get_8_block_model(activation):
    model = Sequential([
        # Block 1
        ZeroPadding2D(input_shape=(32, 32, 3)),  # 34
        Conv2D(384, 3, kernel_regularizer=l2(l=0.0005)),  # 32
        Activation(activation),
        MaxPooling2D(),  # 16
        # Block 2
        Conv2D(384, 1, kernel_regularizer=l2(l=0.0005)),  # 16
        Activation(activation),
        ZeroPadding2D(),  # 18
        Conv2D(384, 2, kernel_regularizer=l2(l=0.0005)),  # 17
        Activation(activation),
        ZeroPadding2D(),  # 19
        Conv2D(640, 2, kernel_regularizer=l2(l=0.0005)),  # 18
        Activation(activation),
        MaxPooling2D(),  # 9
        Dropout(0.1),
        # Block 3
        Conv2D(640, 1, kernel_regularizer=l2(l=0.0005)),  # 9
        Activation(activation),
        ZeroPadding2D(),  # 11
        Conv2D(768, 2, kernel_regularizer=l2(l=0.0005)),  # 10
        Activation(activation),
        ZeroPadding2D(),  # 12
        Conv2D(768, 2, kernel_regularizer=l2(l=0.0005)),  # 11
        Activation(activation),
        ZeroPadding2D(),  # 13
        Conv2D(768, 2, kernel_regularizer=l2(l=0.0005)),  # 12
        Activation(activation),
        MaxPooling2D(),  # 6
        Dropout(0.2),
        # Block 4
        Conv2D(768, 1, kernel_regularizer=l2(l=0.0005)),  # 6
        Activation(activation),
        ZeroPadding2D(),  # 8
        Conv2D(896, 2, kernel_regularizer=l2(l=0.0005)),  # 7
        Activation(activation),
        ZeroPadding2D(),  # 9
        Conv2D(896, 2, kernel_regularizer=l2(l=0.0005)),  # 8
        Activation(activation),
        MaxPooling2D(),  # 4
        Dropout(0.3),
        # Block 5
        Conv2D(896, 1, kernel_regularizer=l2(l=0.0005)),  # 4
        Activation(activation),
        ZeroPadding2D(),  # 6
        Conv2D(1024, 2, kernel_regularizer=l2(l=0.0005)),  # 5
        Activation(activation),
        ZeroPadding2D(),  # 7
        Conv2D(1024, 2, kernel_regularizer=l2(l=0.0005)),  # 6
        Activation(activation),
        MaxPooling2D(),  # 3
        Dropout(0.4),
        # Block 6
        Conv2D(1024, 1, kernel_regularizer=l2(l=0.0005)),  # 3
        Activation(activation),
        Conv2D(1152, 2, kernel_regularizer=l2(l=0.0005)),  # 2
        Activation(activation),
        MaxPooling2D(),  # 1
        Dropout(0.5),
        # Block 7
        Conv2D(1152, 1, kernel_regularizer=l2(l=0.0005)),  # 1
        Activation(activation),
        Dropout(0.5),
        # Block 8
        Conv2D(100, 1, kernel_regularizer=l2(l=0.0005)),  # 1
        Activation('softmax'),
        Flatten(),
    ])
    return model


def get_7_block_model(activation):
    model = Sequential([
        # Block 1
        ZeroPadding2D(padding=(4, 4), input_shape=(32, 32, 3)),  # 40
        Conv2D(192, 5, kernel_regularizer=l2(l=0.0005)),  # 36
        Activation(activation),
        MaxPooling2D(),  # 18
        # Block 2
        Conv2D(192, 1, kernel_regularizer=l2(l=0.0005)),  # 18
        Activation(activation),
        ZeroPadding2D(),  # 20
        Conv2D(240, 3, kernel_regularizer=l2(l=0.0005)),  # 18
        Activation(activation),
        MaxPooling2D(),  # 9
        Dropout(0.1),
        # Block 3
        Conv2D(240, 1, kernel_regularizer=l2(l=0.0005)),  # 9
        Activation(activation),
        ZeroPadding2D(),  # 11
        Conv2D(260, 2, kernel_regularizer=l2(l=0.0005)),  # 10
        Activation(activation),
        MaxPooling2D(),  # 5
        Dropout(0.2),
        # Block 4
        Conv2D(260, 1, kernel_regularizer=l2(l=0.0005)),  # 5
        Activation(activation),
        ZeroPadding2D(),  # 7
        Conv2D(280, 2, kernel_regularizer=l2(l=0.0005)),  # 6
        Activation(activation),
        MaxPooling2D(),  # 3
        Dropout(0.3),
        # Block 5
        Conv2D(280, 1, kernel_regularizer=l2(l=0.0005)),  # 3
        Activation(activation),
        Conv2D(300, 2, kernel_regularizer=l2(l=0.0005)),  # 2
        Activation(activation),
        MaxPooling2D(),  # 1
        Dropout(0.4),
        # Block 6
        Conv2D(300, 1, kernel_regularizer=l2(l=0.0005)),  # 1
        Activation(activation),
        Dropout(0.5),
        # Block 7
        Conv2D(100, 1, kernel_regularizer=l2(l=0.0005)),  # 1
        Activation('softmax'),
        Flatten()
    ])
    return model


def get_7_block_model_rev_2(activation):
    model = Sequential([
        # Block 1
        ZeroPadding2D(input_shape=(32, 32, 3)),  # 34
        Conv2D(384, 3, kernel_regularizer=l2(l=0.0005)),  # 32
        Activation(activation),
        MaxPooling2D(),  # 16
        # Block 2
        Conv2D(384, 1, kernel_regularizer=l2(l=0.0005)),  # 16
        Activation(activation),
        ZeroPadding2D(),  # 18
        Conv2D(384, 2, kernel_regularizer=l2(l=0.0005)),  # 17
        Activation(activation),
        ZeroPadding2D(),  # 19
        Conv2D(640, 2, kernel_regularizer=l2(l=0.0005)),  # 18
        Activation(activation),
        MaxPooling2D(),  # 9
        Dropout(0.1),
        # Block 3
        Conv2D(640, 1, kernel_regularizer=l2(l=0.0005)),  # 9
        Activation(activation),
        ZeroPadding2D(),  # 11
        Conv2D(768, 2, kernel_regularizer=l2(l=0.0005)),  # 10
        Activation(activation),
        ZeroPadding2D(),  # 12
        Conv2D(768, 2, kernel_regularizer=l2(l=0.0005)),  # 11
        Activation(activation),
        ZeroPadding2D(),  # 13
        Conv2D(768, 2, kernel_regularizer=l2(l=0.0005)),  # 12
        Activation(activation),
        MaxPooling2D(),  # 6
        Dropout(0.2),
        # Block 4
        Conv2D(768, 1, kernel_regularizer=l2(l=0.0005)),  # 6
        Activation(activation),
        ZeroPadding2D(),  # 8
        Conv2D(896, 2, kernel_regularizer=l2(l=0.0005)),  # 7
        Activation(activation),
        ZeroPadding2D(),  # 9
        Conv2D(896, 2, kernel_regularizer=l2(l=0.0005)),  # 8
        Activation(activation),
        MaxPooling2D(),  # 4
        Dropout(0.3),
        # Block 5
        Conv2D(896, 1, kernel_regularizer=l2(l=0.0005)),  # 4
        Activation(activation),
        ZeroPadding2D(),  # 6
        Conv2D(1024, 2, kernel_regularizer=l2(l=0.0005)),  # 5
        Activation(activation),
        ZeroPadding2D(),  # 7
        Conv2D(1024, 2, kernel_regularizer=l2(l=0.0005)),  # 6
        Activation(activation),
        MaxPooling2D(),  # 3
        Dropout(0.4),
        # Block 6
        Conv2D(1024, 1, kernel_regularizer=l2(l=0.0005)),  # 3
        Activation(activation),
        Conv2D(1152, 2, kernel_regularizer=l2(l=0.0005)),  # 2
        Activation(activation),
        MaxPooling2D(),  # 1
        Dropout(0.5),
        # Block 7
        Conv2D(1152, 1, kernel_regularizer=l2(l=0.0005)),  # 1
        Activation(activation),
        Dropout(0.5),
        Conv2D(100, 1, kernel_regularizer=l2(l=0.0005)),  # 1
        Activation('softmax'),
        Flatten()
    ])
    return model


if __name__ == '__main__':
    # get data
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = np.pad(x_train, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant')
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # get datagen
    datagen = get_datagen(x_train)
    # get model
    model = get_7_block_model_rev_2('elu')
    # get lr schedule
    schedule = get_lr_schedule()
    # get tensorboard
    tensorboard = get_tensorboard('cifar100/{}-7blocks-elu'.format(datetime.datetime.now()))
    # compile model
    model.compile(SGD(lr=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_k_categorical_accuracy])
    # fit model
    hist = model.fit_generator(crop_generator(datagen.flow(x_train, y_train, batch_size=50), 32),
                               validation_data=crop_generator(datagen.flow(x_test, y_test, batch_size=100), 32),
                               steps_per_epoch=1000,
                               validation_steps=100,
                               epochs=165,
                               callbacks=[tensorboard, schedule]
                               )
