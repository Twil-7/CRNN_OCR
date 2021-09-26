import numpy as np
import cv2
import os


def read_path():

    data_x = []
    data_y = []

    filename = os.listdir('img')
    filename.sort()
    for name in filename:

        img_path = 'img/' + name
        data_x.append(img_path)

        obj1 = name.split('.')
        obj2 = obj1[0].split('_')
        obj3 = obj2[1]
        data_y.append(obj3)

    return data_x, data_y


def make_data():

    data_x, data_y = read_path()
    print('all image quantity : ', len(data_y))    # 10000

    train_x = data_x[:8000]
    train_y = data_y[:8000]
    val_x = data_x[8000:9000]
    val_y = data_y[8000:9000]
    test_x = data_x[9000:]
    test_y = data_y[9000:]

    return train_x, train_y, val_x, val_y, test_x, test_y

