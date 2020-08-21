import torch
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
from common_tools import transform_invert

val_interval = 1
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 均为灰度图像，只需要转换为tensor
x_transforms = transforms.ToTensor()
y_transforms = transforms.ToTensor()

train_curve = list()
valid_curve = list()


def train_model(model, criterion, optimizer, dataload, num_epochs=100):
    model_path = "./model/weights_20.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        start_epoch = 20
        print('加载成功！')
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')

    for epoch in range(start_epoch+1, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_curve.append(loss.item())
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), './model/weights_%d.pth' % (epoch + 1))

        # Validate the model
        valid_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
        valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)
        if (epoch + 2) % val_interval == 0:
            loss_val = 0.
            model.eval()
            with torch.no_grad():
                step_val = 0
                for x, y in valid_loader:
                    step_val += 1
                    x = x.type(torch.FloatTensor)
                    inputs = x.to(device)
                    labels = y.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss_val += loss.item()

                valid_curve.append(loss_val)
                print("epoch %d valid_loss:%0.3f" % (epoch, loss_val / step_val))

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(dataload)
    valid_x = np.arange(1, len(
        valid_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是EpochLoss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()
    return model


# 训练模型
def train(args):
    model = Unet(1, 1).to(device)
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("./data/train", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)


# 显示模型的输出结果
def test(args):
    model = Unet(1, 1)
    model.load_state_dict(torch.load(args.ckpt, map_location='cuda'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)

    save_root = './data/predict'

    model.eval()
    plt.ion()
    index = 0
    with torch.no_grad():
        for x, ground in dataloaders:
            x = x.type(torch.FloatTensor)
            y = model(x)
            x = torch.squeeze(x)
            x = x.unsqueeze(0)
            ground = torch.squeeze(ground)
            ground = ground.unsqueeze(0)
            img_ground = transform_invert(ground, y_transforms)
            img_x = transform_invert(x, x_transforms)
            img_y = torch.squeeze(y).numpy()
            # cv2.imshow('img', img_y)
            src_path = os.path.join(save_root, "predict_%d_s.png" % index)
            save_path = os.path.join(save_root, "predict_%d_o.png" % index)
            ground_path = os.path.join(save_root, "predict_%d_g.png" % index)
            img_ground.save(ground_path)
            # img_x.save(src_path)
            cv2.imwrite(save_path, img_y * 255)
            index = index + 1
            # plt.imshow(img_y)
            # plt.pause(0.5)
        # plt.show()


# 计算Dice系数
def dice_calc(args):
    root = './data/predict'
    nums = len(os.listdir(root)) // 3
    dice = list()
    dice_mean = 0
    for i in range(nums):
        ground_path = os.path.join(root, "predict_%d_g.png" % i)
        predict_path = os.path.join(root, "predict_%d_o.png" % i)
        img_ground = cv2.imread(ground_path)
        img_predict = cv2.imread(predict_path)
        intersec = 0
        x = 0
        y = 0
        for w in range(256):
            for h in range(256):
                intersec += img_ground.item(w, h, 1) * img_predict.item(w, h, 1) / (255 * 255)
                x += img_ground.item(w, h, 1) / 255
                y += img_predict.item(w, h, 1) / 255
        if x + y == 0:
            current_dice = 1
        else:
            current_dice = round(2 * intersec / (x + y), 3)
        dice_mean += current_dice
        dice.append(current_dice)
    dice_mean /= len(dice)
    print(dice)
    print(round(dice_mean, 3))


if __name__ == '__main__':
    #参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train, test or dice", default="train")
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file", default="./model/weights_20.pth")
    # parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
    elif args.action == "dice":
        dice_calc(args)