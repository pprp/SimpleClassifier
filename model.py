import torch
import torch.nn as nn
from torchvision import models
from torch.nn import init
from config import cfg
import os

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 class_num,
                 dropout=False,
                 relu=False,
                 num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        #add_block += [nn.Linear(input_dim, num_bottleneck)]
        num_bottleneck = input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        f = self.add_block(x)
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(f)
        return x


class BasicModule(nn.Module):
    def __init__(self, in_channel, out_channel, bn=True, relu=True):
        super(BasicModule, self).__init__()
        self.bn_swich = bn
        self.relu_switch = relu
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.bn_swich:
            x = self.bn(x)
        if self.relu_switch:
            x = self.relu(x)
        x = self.conv(x)
        return x


class SimpleConv(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleConv, self).__init__()
        self.basic1 = BasicModule(3, 6)
        self.maxpool1 = nn.MaxPool2d(2)
        self.basic2 = BasicModule(6, 12)
        self.maxpool2 = nn.MaxPool2d(2)
        self.basic3 = BasicModule(12, 24)
        self.maxpool3 = nn.MaxPool2d(2)
        self.basic4 = BasicModule(24, 36)
        self.maxpool4 = nn.MaxPool2d(2)

        # dual pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Linear(72, num_classes)

    def forward(self, x):
        x = self.basic1(x)
        x = self.maxpool1(x)
        x = self.basic2(x)
        x = self.maxpool2(x)
        x = self.basic3(x)
        x = self.maxpool3(x)
        x = self.basic4(x)
        x = self.maxpool4(x)
        # dual pooling
        x1 = self.maxpool(x)
        x2 = self.avgpool(x)
        x = torch.cat([x1, x2], dim=1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            folder = os.path.join('weights', cfg.SAVE_FOLDER_NAME)
            if not os.path.exists(folder):
                os.mkdir(folder)
            import time
            name = time.strftime(folder+'/'+'%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name


class DenseConv(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseConv, self).__init__()
        dense = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(dense.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


class DenseConvWithDropout(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseConvWithDropout, self).__init__()
        dense = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(dense.children())[:-1])
        self.drop = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = torch.squeeze(x)
        x = self.drop(x)
        x = self.classifier(x)
        return x


class DualRes18(nn.Module):
    def __init__(self, num_classes=10):
        super(DualRes18, self).__init__()
        res = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(res.children())[:-3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        x = torch.cat([x1, x2], dim=1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


class DualRes50(nn.Module):
    def __init__(self, num_classes=10):
        super(DualRes50, self).__init__()
        res = models.resnet50(pretrained=False)
        self.base = nn.Sequential(*list(res.children())[:-3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)
        #  ClassBlock(1024,
        #              num_classes,
        #              dropout=True,
        #              relu=True,
        #              num_bottleneck=256)

    def forward(self, x):
        x = self.base(x)
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        x = torch.add(x1, x2)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = DualRes50(10)
    print(model)
    ins = torch.FloatTensor(torch.zeros([2, 3, 224, 224]))
    outs = model(ins)
    print(outs.shape)