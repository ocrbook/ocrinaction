# coding=utf-8
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn

# 训练好的模型参数，如./data/crnn.pth
model_path = ''
# 需要测试的图片，如./data/demo.png
img_path = ''
# 和类别对应的字母表，如 0123456789abcdefghijklmnopqrstuvwxyz
alphabet = ''

# imgH(图片高度), nc(图片通道数), nclass(类别数), nh(隐层数)
model = crnn.CRNN(
    imgH=32,
    nc=1,
    nclass=len(alphabet) + 1,
    nh=256
)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

# 图片处理
transformer = dataset.resizeNormalize((192, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

# 输入模型并预测结果
model.eval()
preds = model(image)
_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

# 对结果进行解码，类别数和字母表中字符对应，并输出结果
preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
