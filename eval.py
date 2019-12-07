import torch
import os
import argparse

from dataloader import valid_dataloader, valid_dataset
from config import cfg
from model import SimpleConv, DenseConv
from train import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser("locate weight")
    parser.add_argument("--weight",
                        type=str,
                        default="./weights/model1/dense121_2019_12_7_10.pth")
    args = parser.parse_args()

    model = DenseConv(cfg.NUM_CLASSES)

    if args.weight is not "":
        model.load_state_dict(torch.load(args.weight))

    if torch.cuda.is_available():
        model = model.cuda()

    # model.eval()
    # sum_correct = 0
    # for image, label in valid_dataloader:
    #     if torch.cuda.is_available():
    #         image, label = image.cuda(), label.cuda()
    #     out = model(image)
    #     prediction = torch.max(out, 1)[1]
    #     correct = (prediction == label).sum()
    #     sum_correct += correct
    # test_acc = sum_correct.float() / len(valid_dataset)
    test_acc = test(model)
    print("Accuracy of %s is %.3f" % (os.path.basename(args.weight), test_acc))
