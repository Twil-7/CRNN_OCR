import numpy as np
import cv2
from c_rnn_model import create_c_rnn
import itertools
import os
from get_data import make_data

# ctc算法的blank机制，在原有类别基础上多添加一个空白分类
ctc_blank = 10
class_dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
class_list = list(class_dictionary.keys())
img_w = 128
img_h = 64
char_num = 6


def predict_test_data(test_x, test_y):

    c_rnn_model = create_c_rnn(loss_model=False)
    c_rnn_model.summary()
    c_rnn_model.load_weights('best_weights_0.090.h5')

    print('total test quantity : ', len(test_x))
    accuracy_count = 0
    for i in range(len(test_x)):

        img = cv2.imread(test_x[i])
        img1 = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
        img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img3 = img2 / 255
        img4 = img3[np.newaxis, :, :, np.newaxis]

        out1 = c_rnn_model.predict(img4)                    # out.shape : (1, 32, 11)
        out2 = np.argmax(out1[0, 2:], axis=1)               # get max index -> len = 32
        out3 = [k for k, g in itertools.groupby(out2)]      # remove overlap value
        out4 = ''

        for j in range(len(out3)):
            index = int(out3[j])
            if index < ctc_blank:
                plate_char = class_list[index]
                out4 = out4 + str(plate_char)

        y_pre = out4
        y_true = test_y[i]

        if y_pre == y_true:
            accuracy_count = accuracy_count + 1
        else:
            print('算法识别错误 : ', test_x[i])
            print('y_pre : ', y_pre)
            print('y_true :', y_true)
            # cv2.namedWindow("Image")
            # cv2.imshow("Image", img3)
            # cv2.waitKey(0)

    print('The test accuracy is : ', accuracy_count/len(test_x))

# The test accuracy is :  0.973
