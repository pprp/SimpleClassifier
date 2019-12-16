import torch
import os
import torch.nn as nn
import torch.optim as optim
import argparse

from dataloader import train_dataloader, valid_dataloader, valid_dataset
from config import cfg
from model import SimpleConv, DenseConv, DualRes18, DenseConvWithDropout
from utils.visdom import Visualizer
import model


def train(model, optimizer, criterion, scheduler):
    model.train()
    vis = Visualizer(env='main')

    for epoch in range(cfg.MAX_EPOCH):
        batch = 0
        for num, (image_4dArray, label_2dArray) in enumerate(train_dataloader):
            batch_loss = 0
            train_acc = 0

            if torch.cuda.is_available():
                image_4dArray, label_2dArray = image_4dArray.cuda(
                ), label_2dArray.cuda()

            out = model(image_4dArray)
            loss = criterion(out, label_2dArray)

            batch_loss = loss.item()
            prediction = torch.max(out, 1)[1]

            train_correct = (prediction == label_2dArray).sum()

            train_acc = train_correct.float() / cfg.BATCH_SIZE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch += 1
            if num % 10 == 0 and num > 0:
                print(
                    "Epoch: %d/%d || batch:%03d/%02d || batch_loss: %.3f || train_acc: %.3f || lr: %.7f"
                    % (epoch, cfg.MAX_EPOCH, batch, len(train_dataloader),
                       batch_loss, train_acc, optimizer.param_groups[0]['lr']))
                vis.plot_many_stack({'batch_loss': batch_loss})
                vis.plot_many_stack({"train_acc": train_acc.item()})
        # test
        test_acc = test(model)
        print("Epoch: %d || test_acc: %.3f" % (epoch, test_acc))
        vis.plot_many_stack({"test_acc": test_acc.item()})

        if epoch % 5 == 0 and epoch > 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_folder,
                             cfg.SAVE_WEIGHT_NAME + "_" + str(epoch) + ".pth"))
        scheduler.step()


def test(model):
    model.eval()
    sum_correct = 0.0
    for image, label in valid_dataloader:
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        output = model(image)
        # print(output.shape)
        prediction = torch.max(output, 1)[1]
        correct = (prediction == label).sum()
        sum_correct += correct
    test_acc = sum_correct.float() / len(valid_dataset)
    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Simple Conv")
    parser.add_argument('--epoch',
                        type=int,
                        default=120,
                        help="max epoch for training")
    parser.add_argument('--gpu', type=str, default="0", help="use which gpu")

    arg = parser.parse_args()

    # save weigth to save folder
    save_folder = os.path.join("./weights", cfg.SAVE_FOLDER_NAME)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # make basic model
    # from model import SimpleConv, DenseConv, DualRes18, DenseConvWithDropout
    # model = DenseConvWithDropout(cfg.NUM_CLASSES)
    model = getattr('model', 'SimpleConv')(cfg.NUM_CLASSES)

    # load weight
    pretrained_path = cfg.PRETRAINED_MODEL_PATH
    if pretrained_path is not "":
        model.load_state_dict(torch.load(pretrained_path))

    # cuda config
    if torch.cuda.is_available():
        model = model.cuda()

    # optimier loss function and  scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train(model, optimizer, criterion, scheduler)