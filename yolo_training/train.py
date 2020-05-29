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

import warnings


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/museum/images", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="cfg/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="cfg/museum.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="cfg/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="cfg/museum.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=600, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
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

# model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

test_set = torch.utils.data.DataLoader(
    ListDataset(valid_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_ap = 0
best_tot_loss = 100000
patience = 5
epoch_counter = 0

with open("train_log__LR_00{}.csv".format(int(learning_rate * 1000)), "w") as f:
    for epoch in range(opt.epochs):

        #print(f"Epoch {epoch + 1}/{opt.epochs} train acc.: {eval_acc(model, dataloader, device):.3f} "
        #      f"test acc.: {eval_acc(model, test_set, device):.3f}\n")

        start = time.time()

        losses = {"x": 0, "y": 0, "w": 0, "h": 0, "conf": 0, "cls": 0, "total": 0, "recall": 0, "precision": 0}
        test_losses = {"x": 0, "y": 0, "w": 0, "h": 0, "conf": 0, "cls": 0, "total": 0, "recall": 0, "precision": 0}
        lst_recall = []
        lst_precision = []

        save_weights = False

        model.train()

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

            losses["x"] += model.losses["x"]
            losses["y"] += model.losses["y"]
            losses["w"] += model.losses["w"]
            losses["h"] += model.losses["h"]
            losses["conf"] += model.losses["conf"]
            losses["cls"] += model.losses["cls"]
            losses["total"] += loss.item()
            losses["recall"] += model.losses["recall"]
            losses["precision"] += model.losses["precision"]

            #lst_recall.append(losses["recall"])
            #lst_precision.append(losses["precision"])

            model.seen += imgs.size(0)

        print('\nEpoch {}/{} Time: {} min'.format(epoch + 1, opt.epochs, (time.time() - start) / 60))

        losses["x"] /= len(dataloader)
        losses["y"] /= len(dataloader)
        losses["w"] /= len(dataloader)
        losses["h"] /= len(dataloader)
        losses["conf"] /= len(dataloader)
        losses["cls"] /= len(dataloader)
        losses["total"] /= len(dataloader)
        losses["recall"] /= len(dataloader)
        losses["precision"] /= len(dataloader)

        print(
            "[Epoch %d/%d] [AVERAGE Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch + 1,
                opt.epochs,
                losses["x"],
                losses["y"],
                losses["w"],
                losses["h"],
                losses["conf"],
                losses["cls"],
                losses["total"],
                losses["recall"],
                losses["precision"],
            )
        )

        # evaluation mode
        model.eval()
        for batch_i, (_, imgs, targets) in enumerate(test_set):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)

            loss_test = model(imgs, targets)

            print(
                "[TEST Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch + 1,
                    opt.epochs,
                    batch_i + 1,
                    len(test_set),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss_test.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )

            test_losses["x"] += model.losses["x"]
            test_losses["y"] += model.losses["y"]
            test_losses["w"] += model.losses["w"]
            test_losses["h"] += model.losses["h"]
            test_losses["conf"] += model.losses["conf"]
            test_losses["cls"] += model.losses["cls"]
            test_losses["total"] += loss_test.item()
            test_losses["recall"] += model.losses["recall"]
            test_losses["precision"] += model.losses["precision"]

            lst_recall.append(test_losses["recall"])
            lst_precision.append(test_losses["precision"])

        test_losses["x"] /= len(test_set)
        test_losses["y"] /= len(test_set)
        test_losses["w"] /= len(test_set)
        test_losses["h"] /= len(test_set)
        test_losses["conf"] /= len(test_set)
        test_losses["cls"] /= len(test_set)
        test_losses["total"] /= len(test_set)
        test_losses["recall"] /= len(test_set)
        test_losses["precision"] /= len(test_set)

        print(
            "[TEST Epoch %d/%d] [AVERAGE Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch + 1,
                opt.epochs,
                test_losses["x"],
                test_losses["y"],
                test_losses["w"],
                test_losses["h"],
                test_losses["conf"],
                test_losses["cls"],
                test_losses["total"],
                test_losses["recall"],
                test_losses["precision"],
            )
        )

        avg_precision = compute_ap(lst_recall, lst_precision)
        print("[TEST Epoch {}/{}] AVG PRECISION = {}\n".format(epoch + 1, opt.epochs, avg_precision))

        # Formato File csv per grafici
        # lossX, lossY, lossW, lossH, lossConf, lossCls, lossTotal, recall, precision, AP
        row = "{},{},{},{},{},{},{},{},{},{}\n".format(test_losses["x"], test_losses["y"], test_losses["w"], test_losses["h"],
                                                          test_losses["conf"], test_losses["cls"], test_losses["total"], test_losses["recall"],
                                                          test_losses["precision"], avg_precision)
        f.write(row)

        #print(f"Epoch {epoch + 1}/{opt.epochs} train acc.: {eval_acc(model, dataloader, device):.3f} "
        #     f"test acc.: {eval_acc(model, test_set, device):.3f}\n")

        if avg_precision > best_ap:
            best_ap = avg_precision
            save_weights = True

        #if (epoch + 1) % opt.checkpoint_interval == 0:
        if save_weights:
            model.save_weights("%s/%d__AP_%d__LR_00%d.weights" % (opt.checkpoint_dir, epoch + 1, int(np.round(avg_precision)), int(learning_rate * 1000)))

        # control when to exit
        if test_losses["total"] < best_tot_loss:
            epoch_counter = 0
        else:
            epoch_counter += 1
            if epoch_counter == patience:
                break

f.close()
