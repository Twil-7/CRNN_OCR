import cv2
import os
import random
import numpy as np
from tensorflow.keras.utils import Sequence
import math
from c_rnn_model import create_c_rnn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


class_dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
img_w = 128
img_h = 64
char_num = 6
w_down_sample = 4    # w方向尺寸缩减了4倍，h方向尺寸缩减了16倍


class SequenceData(Sequence):

    def __init__(self, data_x, data_y, batch_size):
        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.indexes = np.arange(len(self.data_x))

    def __len__(self):
        return math.floor(len(self.data_x) / float(self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    # 训练过程中由于调用了ctc内部算法，部分数据编码处理已被封装，不需要我们再做转换。
    # x_data :(batch, 128, 64, 1)，每一个维度存储图片矩阵信息。
    # y_data :(batch, 6)，每一个维度仅仅是个向量，用0-9来记录每个字符类别信息，而不需要我们专门转化为one-hot编码形式。

    def __getitem__(self, idx):

        batch_index = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.data_x[k] for k in batch_index]
        batch_y = [self.data_y[k] for k in batch_index]

        x_data = np.ones([self.batch_size, img_h, img_w, 1])                           # (batch, 64, 128, 1)
        y_data = np.ones([self.batch_size, char_num])                                  # (batch, 6)
        input_length = np.ones((self.batch_size, 1)) * (img_w // w_down_sample - 2)    # (batch, 1)， 128 / 4 - 2 = 30
        label_length = np.zeros((self.batch_size, 1))                                  # (batch, 1)

        for i in range(self.batch_size):

            img = cv2.imread(batch_x[i])    # (80, 210, 3)

            img1 = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img3 = img2 / 255
            img4 = img3[:, :, np.newaxis]
            x_data[i, :, :, :] = img4

            text = batch_y[i]
            label = np.zeros(char_num)

            # print(text)
            # cv2.namedWindow("Image")
            # cv2.imshow("Image", img3)
            # cv2.waitKey(0)

            for j in range(char_num):
                c = text[j]
                index = class_dictionary[c]
                label[j] = index

            y_data[i] = label
            label_length[i] = len(label)

        inputs = {
            'the_input': x_data,
            'the_labels': y_data,
            'input_length': input_length,
            'label_length': label_length
            }
        outputs = {'ctc': np.zeros([self.batch_size])}

        return inputs, outputs


# create model and train and save
def train_network(train_generator, validation_generator, epoch):

    model = create_c_rnn(loss_model=True)

    adam = Adam(lr=1e-3, amsgrad=True)
    log_dir = "Logs/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}_{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pre: y_pre}, optimizer=adam)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights('first_weights.hdf5')


def load_network_then_train(train_generator, validation_generator, epoch, input_name, output_name):

    model = create_c_rnn(loss_model=True)
    model.load_weights(input_name)

    adam = Adam(lr=1e-4, amsgrad=True)
    log_dir = "Logs/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}_{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pre: y_pre}, optimizer=adam)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights(output_name)
