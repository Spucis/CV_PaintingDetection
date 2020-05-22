from __future__ import division

from yolo_training.models import *
from yolo_training.utils.utils import *
from yolo_training.utils.datasets import *
from yolo_training.utils.parse_config import *

import os
import argparse
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/museum/images", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="cfg/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="cfg/museum.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="cfg/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="cfg/museum.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]
valid_path = data_config["valid"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
#model.apply(weights_init_normal)

if cuda:
    torch.cuda.empty_cache()
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

test_set = torch.utils.data.DataLoader(
    ListDataset(valid_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

#acc = Accuracy()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: PROVA CON IMG IN .JPG (le img in png sono tutte uguali con valore 0.5020)

for epoch in range(opt.epochs):

    #print(f"Epoch {epoch + 1}/{opt.epochs} train acc.: {eval_acc(model, dataloader, device):.3f} "
    #      f"test acc.: {eval_acc(model, test_set, device):.3f}\n")

    start = time.time()

    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch + 1,
                opt.epochs,
                batch_i + 1,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    print('\nEpoch {}/{} Time: {}s'.format(epoch, opt.epochs, time.time() - start))

    #print(f"Epoch {epoch + 1}/{opt.epochs} train acc.: {eval_acc(model, dataloader, device):.3f} "
    #     f"test acc.: {eval_acc(model, test_set, device):.3f}\n")

    if epoch % opt.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
