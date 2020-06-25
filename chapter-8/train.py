# coding=utf-8
from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np

import os

import utils
import dataset

import models.crnn as crnn

parser = argparse.ArgumentParser()
# 指定训练集根目录
parser.add_argument('--trainroot', required=True, help='训练集路径')
# 指定验证集根目录
parser.add_argument('--valroot', required=True, help='验证集路径')

parser.add_argument('--workers', type=int, help='提取数据的线程数', default=0)
# 指定每次输入的图片数量，默认为16张
parser.add_argument('--batchSize', type=int, default=16, help='输入批次数量')

# 指定输入图片高度，默认为32个像素
parser.add_argument('--imgH', type=int, default=32, help='输入图像高度，默认为32像素')
# 指定输入图片高度，默认为100个像素
parser.add_argument('--imgW', type=int, default=192, help='输入图像宽度，默认为192像素')
parser.add_argument('--nh', type=int, default=128, help='LSTM隐层单元数')
parser.add_argument('--nepoch', type=int, default=500, help='需要训练的轮数，默认为500轮')

# 是否使用GPU进行训练
parser.add_argument('--cuda', action='store_true', help='使用GPU加速')
parser.add_argument('--ngpu', type=int, default=1, help='使用GPU的个数')
parser.add_argument('--pretrained', default='', help="预训练参数路径")

parser.add_argument('--expr_dir', default='expr', help='保存参数的路径位置')
parser.add_argument('--n_test_disp', type=int, default=1, help='进行测试时显示的样本数')
parser.add_argument('--saveInterval', type=int, default=500, help='隔多少迭代显示一次')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--beta1', type=float, default=0.5, help='adam优化器的beta1参数')
parser.add_argument('--adam', action='store_true', help='是否是用adam优化器 (默认为rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='是否是用adadelta优化器 (默认为rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='图片修改尺寸时保持长宽比')
parser.add_argument('--manualSeed', type=int, default=4321, help='')
parser.add_argument('--random_sample', action='store_true', help='是否随机采样')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.RawDataset(gt_file=os.path.join(opt.trainroot, 'gt.txt'), gt_dir=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=None,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.RawDataset(
    gt_file=os.path.join(opt.trainroot, 'gt.txt'), gt_dir=opt.trainroot, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))

nclass = len(train_dataset.lexicon) + 1
nc = 1

converter = utils.strLabelConverter(train_dataset.lexicon)
criterion = torch.nn.CTCLoss(blank=0, reduction='none')


# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    crnn.load_state_dict(torch.load(opt.pretrained))
print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        #preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):

    data = train_iter.next()

    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size)).cuda()
    cost = criterion(preds, text.long(), preds_size.long(), length.long()).sum() / float(batch_size)

    # cost = cost

    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0

    while i < len(train_loader):

        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), cost))

        # 保存模型
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/CRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))

    val(crnn, test_dataset, criterion)

