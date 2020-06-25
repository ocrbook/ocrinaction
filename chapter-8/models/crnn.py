# coding=utf-8
import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        # 图片高度H必须被16整除
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        # ks: 卷积核尺寸
        ks = [3, 3, 3, 3, 3, 3, 2]

        # ps: 补全尺寸
        ps = [1, 1, 1, 1, 1, 1, 0]

        # ks: 卷积步长
        ss = [1, 1, 1, 1, 1, 1, 1]

        # nm: 通道数
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv_relu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        conv_relu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nclass),)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()

        # 确保卷积特征高度为1，从而能够转化为rnn输入格式
        assert h == 1, "the height of conv must be 1"

        # LSTM接受的输入格式为 (n(样本数), t(时间步), c(通道数))
        # 因此需要将卷积的特征进行permute变换: (n, c, h, w) -> (t, n, c)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        # rnn features
        output = self.rnn(conv)

        return output
