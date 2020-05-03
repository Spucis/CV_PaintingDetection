from source import globals
from source.detection_utils import *
import json
import os

conf = globals.conf

class PaintingManager:
    def __init__(self, video_manager):
        self.video_manager = video_manager
        self.input_path = conf['input_path'] + conf['in_dir'] + conf['slash']
        self.input_img = conf['input_img']
        self.count = 0
        self.cap = None
        self.out = None
        self.inodes = None
        self.kp_dict = {}
        self.des_dict = {}

    def open_video(self, video_name):
        self.cap, self.out = self.video_manager.open_video(video_name, self.input_path, conf['output_path'])

    def close_video(self):
        self.cap.release()
        self.out.release()

    def ROI_detection(self, or_frame):

        gray_frame, marked_frame, ed_frame = edge_detection(or_frame, debug=True,  frame_number=self.count)
        #kp_frame = keypoints_detection(or_frame, show=False)
        roi_frame, ROIs = ccl_detection(or_frame, gray_frame, ed_frame, frame_number=self.count)

        #ed_frame = cv2.cvtColor(ed_frame, cv2.COLOR_GRAY2BGR)

        return roi_frame, ROIs

    def paint_detection(self):
        # Read until video is completed
        while (self.cap.isOpened() and self.out.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                if self.count == 0:
                    print("Edge detection function.")
                mod_frame, ROIs = self.ROI_detection(frame)
                # hough_transform()
                self.count += 1
                if self.count % 100 == 0:
                    print("Frame count: {}/{}".format(self.count, self.video_manager.n_frame))
                self.out.write(mod_frame)


                all_video = True
                if not all_video:
                    break
            else:
                print("Problema durante la lettura del video... stop al frame {}: {}".format(self.count, frame))
                break
        print("Fine edge_detection.")

    def keypoint_writedb(self):
        self.inodes = os.listdir(self.input_img)
        self.inodes.remove('.gitkeep')
        self.inodes.sort()

        for i in self.inodes:
            img = cv2.imread(self.input_img + i)
            kp, des = find_keypoint(img)

            self.kp_dict[i] = kp
            self.des_dict[i] = des

    def keypoint_readdb(self):
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        des1 = self.des_dict[self.inodes[0]]
        des2 = self.des_dict[self.inodes[2]]

        # Match descriptors.
        matches_d = bf.match(des1, des2)
        matches_u = bf.match(des1, des1)

        # Sort them in the order of their distance.
        matches = sorted(matches_d, key=lambda x:x.distance)

        d_d = []
        d_u = []
        for el in matches_d:
            d_d.append(el.distance)
        for el in matches_u:
            d_u.append(el.distance)

        print("U: -> " + str(max(d_u)))
        print("D: -> " + str(max(d_d)))

        img1 = cv2.imread(self.input_img + self.inodes[0])
        img2 = cv2.imread(self.input_img + self.inodes[1])

        kp1 = self.kp_dict[self.inodes[0]]
        kp2 = self.kp_dict[self.inodes[1]]

        # Draw first 10 matches.
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:], flags=2, outImg=None)

        d_img = {}
        d_img['matcher'] = img3
        show_frame(d_img)
