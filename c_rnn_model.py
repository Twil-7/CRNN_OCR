from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.python.keras.layers.cudnn_recurrent import CuDNNGRU
from tensorflow.keras import backend as k
from tensorflow.python.keras.layers.recurrent import LSTM


# 模型在高度方向进行了5次下采样，在宽度方向进行了2次下采样，原始图片的高度缩小了32倍，宽度缩小了4倍
# 输入尺寸为(None, 64, 128, 3)，输出feature map尺寸为(None, 2, 32, 256)。

def feature_extractor(x):

    # 5次下采样卷积
    for i in range(5):
        # 在下采样卷积之前，先执行两次卷积运算
        for j in range(2):

            x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)

        x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3,
                   strides=2 if i < 2 else (2, 1), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

    return x


# Permute层: 根据给定的dim置换输入的维度, 根据指定的模式重新排列。输入数据尺寸由(None, 2, 32, 256)转变为(None, 32, 2, 256)。
# TimeDistributed函数: 实现了输入数据从三维到二维的转换，将输入数据的最后两个维度展开, 只将最后两个维度所有数据拉直合并。

# 经过双向RNN层处理，所输出的形状为(None, 32, 512)
# 其中32代表序列长度为32，256代表每个cell都是一个向量，由正反两个方向各256个单元所输出的结果组成。

def rnn_feature(x):

    x = Permute((2, 1, 3))(x)    # 交换维度
    x = TimeDistributed(Flatten())(x)

    lstm_1 = LSTM(256, return_sequences=True, name='lstm1', kernel_initializer='he_normal')(x)
    lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, name='lstm1_b', kernel_initializer='he_normal')(x)
    reversed_lstm_1b = Lambda(lambda lstm_tensor: k.reverse(lstm_tensor, axes=1))(lstm_1b)

    lstm1_merged = add([lstm_1, reversed_lstm_1b])
    lstm1_merged = BatchNormalization()(lstm1_merged)

    lstm_2 = LSTM(256, return_sequences=True, name='lstm2', kernel_initializer='he_normal')(lstm1_merged)
    lstm_2b = LSTM(256, return_sequences=True, go_backwards=True,
                   name='lstm2_b', kernel_initializer='he_normal')(lstm1_merged)
    reversed_lstm_2b = Lambda(lambda lstm_tensor: k.reverse(lstm_tensor, axes=1))(lstm_2b)

    lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])
    lstm2_merged = BatchNormalization()(lstm2_merged)

    return lstm2_merged


def ctc_lambda_func(args):

    y_pre, labels, input_length, label_length = args

    # the 2 is critical here， since the first couple outputs of the RNN tend to be garbage:
    y_pre = y_pre[:, 2:, :]
    return k.ctc_batch_cost(labels, y_pre, input_length, label_length)


def create_c_rnn(loss_model=True):

    # 输入网络中的图片长宽是128，64，共需识别6个字符，加上blank总共分类类别数为10+1

    img_w = 128
    img_h = 64
    char_num = 6
    num_class = 10 + 1

    inputs = Input((img_h, img_w, 1), name='the_input')

    x = feature_extractor(inputs)
    x = rnn_feature(x)

    y_pre = Dense(num_class, activation='softmax')(x)

    # create loss layer
    labels = Input(name='the_labels', shape=[char_num], dtype='float32')    # (None, 6)
    input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
    label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pre, labels, input_length, label_length])
    # (None, 1)

    if loss_model:
        c_rnn_loss = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
        c_rnn_loss.summary()
        return c_rnn_loss
    else:
        c_rnn = Model(inputs=[inputs], outputs=y_pre)
        return c_rnn
