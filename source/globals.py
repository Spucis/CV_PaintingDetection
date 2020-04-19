import numpy as np
import cv2
from json import *

file = open('conf.json', 'r').read()
conf = JSONDecoder().decode(s=file)
print("Config file: \n{}".format(conf))

class VideoManager:
    def __init__(self):
        self.in_codec = None
        self.in_fps = None
        self.in_frameSize = None
        self.n_frame = None


    def print_info(self):
        print("Frame count: {}".format(self.n_frame))
        print("Codec number: {}\nFPS: {}\nFrame size: {}".format(int(self.in_codec),
                                                                 np.around(self.in_fps).astype(np.uint32),
                                                                 np.around(self.in_frameSize).astype(np.uint32)))


    def open_video(self, video_name, input_path, output_path):
        cap = cv2.VideoCapture()
        cap.open("{}{}.MP4".format(input_path, video_name))
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return None, None

        self.in_codec = cap.get(cv2.CAP_PROP_FOURCC)
        self.in_fps = cap.get(cv2.CAP_PROP_FPS)
        self.in_frameSize = np.around((cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))).astype(
            np.uint32)
        self.n_frame = np.around(cap.get(cv2.CAP_PROP_FRAME_COUNT)).astype(np.uint32)

        out = cv2.VideoWriter("{}{}.MP4".format(output_path, video_name), cv2.VideoWriter_fourcc(*'mp4v'),
                              np.around(self.in_fps).astype(np.uint32),
                              tuple(np.around(self.in_frameSize).astype(np.uint32)), True)
        out.open("{}{}.MP4".format(output_path, video_name), cv2.VideoWriter_fourcc(*'mp4v'),
                 np.around(self.in_fps).astype(np.uint32),
                 tuple(np.around(self.in_frameSize).astype(np.uint32)))

        if (out.isOpened() == False):
            print("Error opening out video stream or file")
            return None, None

        return cap, out
