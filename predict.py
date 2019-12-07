import torch
import os
import argparse
import cv2

from model import SimpleConv, DenseConv
from config import cfg


parser = argparse.ArgumentParser("test image path")
parser.add_argument('--image_path', type=str, default="./data/test", help="test image path")
parser.add_argument('--weight',
                    type=str,
                    default='./weights/model1/dense121_2019_12_7_10.pth',
                    help='which weights to load')
args = parser.parse_args()


model = DenseConv(cfg.NUM_CLASSES)
if args.weight is not "":
    model.load_state_dict(torch.load(args.weight))
if torch.cuda.is_available():
    model = model.cuda()

labels2classes = cfg.labels_to_classes

model.eval()

batch_image = []
name_image = []

img_tensor = None
for jpg_name in os.listdir(args.image_path):
    if jpg_name.endswith('.jpg'):
        img = cv2.imread(os.path.join(args.image_path, jpg_name))
        name_image.append(jpg_name)
        img = cv2.resize(img, cfg.INPUT_SIZE)
        img = torch.FloatTensor(img).unsqueeze(0)
        if img_tensor is None:
            img_tensor = img
        else:
            img_tensor = torch.cat([img_tensor, img], dim=0)
        # batch_image.append(img)
        # print(img.shape, img_tensor.shape)

# print(img_tensor.shape)
img_tensor = img_tensor.permute(dims=[0,3,1,2])
# print(img_tensor.shape)

with torch.no_grad():
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    out = model(img_tensor)
    prediction = torch.max(out, 1)[1]
    print(prediction.shape)
    for i,p in enumerate(prediction):
        print(i, name_image[i], "\t", labels2classes[str(p.cpu().numpy())])
    
