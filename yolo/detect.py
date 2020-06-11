from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from yolo.util import *
import argparse
import os 
import os.path as osp
from yolo.darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse(separator = "\\", base_path='yolo'):
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="{}imgs".format(base_path+separator), type=str)
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="{}det".format(base_path+separator), type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.75)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="{}cfg{}yolov3.cfg".format(base_path+separator, separator), type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="{}yolov3.weights".format(base_path+separator), type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


def detect(specific_frame=None, verbose=True, draw_images=False, write_on_disk=False, separator="\\", base_path='yolo', model=None, room=None):
    args = arg_parse(separator=separator, base_path=base_path)
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = False #torch.cuda.is_available()



    num_classes = 80
    classes = load_classes("{}{}data{}coco.names".format(base_path,separator,separator))



    #Set up the neural network

    if verbose:
        print("Loading network.....")

    if not model:
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightsfile)
    if verbose:
        print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()


    #Set the model in evaluation mode
    model.eval()

    read_dir = time.time()
    #Detection phase
    try:
        if specific_frame is not None:
            imlist = ["specific_frame_no_name_im_list"]
        else:
            imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()
    if specific_frame is not None:
        loaded_ims = [specific_frame]
    else:
        loaded_ims = [cv2.imread(x) for x in imlist]

    #if specific_frame is not None: # MARCO (G20). if the specific frame is set, the loaded_ims list is substituted with the single frame
    #    loaded_ims = list(specific_frame)

    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)


    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i*batch_size : min((i + 1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]

    write = 0


    if CUDA:
        im_dim_list = im_dim_list.cuda()

    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):
    #load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

        end = time.time()

        if type(prediction) == int:
            if verbose:
                for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                    im_id = i*batch_size + im_num
                    print("{0:20s} predicted in {1:6.3f} seconds".format(image.split(separator)[-1], (end - start)/batch_size))
                    print("{0:20s} {1:s}".format("Objects Detected:", ""))
                    print("----------------------------------------------------------")
            continue

        prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist

        if not write:                      #If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))

        if verbose:
            for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split(separator)[-1], (end - start)/batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                print("----------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()
    try:
        output
    except NameError:
        if verbose:
            print ("No detections were made")
        return [], loaded_ims # MARCO. No detections, so i return empty list for rois and the loaded_ims
        #exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

    scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)


    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])


    output_recast = time.time()
    class_load = time.time()
    colors = pkl.load(open("{}pallete".format(base_path+separator), "rb"))

    if draw_images:
        draw = time.time()

        bbox_verbose = False
        if verbose and bbox_verbose:
            for index, element in enumerate(output.numpy()):
                print("-------\nImage: {}\nBBox:\n{}\n{}\nClass: {}[{}]".format(imlist[int(element[0])],element[1:3],element[3:5], element[-1], classes[int(element[-1])]))
                #print("..{}".format(element))

        def write(x, results):
            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())
            img = results[int(x[0])]
            cls = int(x[-1])
            room_label = "- Stanza non id." if not room else "- Stanza "+str(room)
            if cls == 0: # only CLASS "PERSON"
                color = random.choice(colors)
                label = "{0} {1}".format(classes[cls], room_label)
                cv2.rectangle(img, c1, c2,color, 1)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(img, c1, c2,color, -1)
                cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
            return img


        list(map(lambda x: write(x, loaded_ims), output))

        det_names = pd.Series(imlist).apply(lambda x: "{}{}det_{}".format(args.det,separator,x.split(separator)[-1]))


        if write_on_disk:
            list(map(cv2.imwrite, det_names, loaded_ims))


    end = time.time()

    if verbose:
        print("SUMMARY")
        print("----------------------------------------------------------")
        print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
        print()
        print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
        print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
        print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
        print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
        print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
        print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
        print("----------------------------------------------------------")


    torch.cuda.empty_cache()

    output_people_ROIs = []
    for index, element in enumerate(output.numpy()):
        if classes[int(element[-1])] == 'person':
            output_people_ROIs.append((element[1:3],element[3:5]))
        """
        print("-------\nImage: {}\nBBox:\n{}\n{}\nClass: {}[{}]".format(imlist[int(element[0])], element[1:3],
                                                                        element[3:5], element[-1],
                                                                        classes[int(element[-1])]))
        # print("..{}".format(element))
        """


    return output_people_ROIs, loaded_ims
