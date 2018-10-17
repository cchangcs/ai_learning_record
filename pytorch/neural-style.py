'''
图像风格：neural-style，将一个内容图像和一个风格图像作为输入，返回一个按照所选择的风格图像加工的内容图像。
定义两个距离，一个用于内容（Dc)，一个用于风格（Ds),Dc测量的是两个图像的内容有多像，Ds测量两个图像的风格有多
像，然后采用一个新图像，对它进行变化，同时最小化与内容图像和风格图像的距离。
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy

# cuda
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# 输出图片的大小
imsize = 512 if use_cuda else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor(),
])


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


style_img = image_loader('images/2.jpg').type(dtype)
content_img = image_loader('images/1.jpg').type(dtype)

assert style_img.size() == content_img.size(), 'we need to import' \
                                               ' style and content images of the same size'

# 显示图片
unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()


def imshow(tensor, title=None):
    image = tensor.clone().cpu()  # clone the tensor to not do changes on it
    image = image.view(3, imsize, imsize)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(2)

plt.figure()
imshow(style_img.data, 'Style Image')
plt.figure()
imshow(content_img.data, 'Content Image')


# Content loss
class ContentLoss(nn.Module):
    def __init__(self, target ,weight):
        super(ContentLoss, self).__init__()
        # 从树中分割出目标内容
        self.target = target.detach() * weight
        # 动态的计算梯度：状态值而不是变量，否则criterion
        # 的前向方法将会返回一个error
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.los.backward(retain_graph=retain_graph)
        return self.loss


# Style  loss
class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a = batch size(=1)
        # b = number of Feature
        # (c, d) = dimensions of
        features = input.view(a * b, c * d)  # resize F_XL into
        G = torch.mm(features, features.t())
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


cnn = models.vgg19(pretrained=True).features

# move it to the GPU if possible
if use_cuda():
    cnn = cnn.cuda()

# desired depth layers to compute style/content losses




