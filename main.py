import numpy as np
import cv2
from get_data import make_data
from c_rnn_model import create_c_rnn
from train import SequenceData
from train import train_network
from train import load_network_then_train
import itertools
from predict import predict_test_data


class_dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}


if __name__ == "__main__":

    train_x, train_y, val_x, val_y, test_x, test_y = make_data()

    train_generator = SequenceData(train_x, train_y, 32)
    test_generator = SequenceData(test_x, test_y, 32)

    # train_network(train_generator, test_generator, epoch=50)
    # load_network_then_train(train_generator, test_generator, epoch=20,
    #                         input_name='first_weights.hdf5', output_name='second_weights.hdf5')

    predict_test_data(test_x, test_y)







