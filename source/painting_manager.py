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
        self.ROIs = []

    def open_video(self, video_name):
        self.cap, self.out = self.video_manager.open_video(video_name, self.input_path, conf['output_path'])

    def close_video(self):
        self.cap.release()
        self.out.release()

    def ROI_detection(self, or_frame):
        gray_frame, marked_frame, ed_frame = edge_detection(or_frame, debug=True,  frame_number=self.count)
        #kp_frame = keypoints_detection(or_frame, show=False)
        ccl_frame, ROIs = ccl_detection(or_frame, gray_frame, ed_frame, frame_number=self.count)

        #ed_frame = cv2.cvtColor(ed_frame, cv2.COLOR_GRAY2BGR)

        return ccl_frame, ROIs

    def paint_detection(self):
        # Read until video is completed
        while (self.cap.isOpened() and self.out.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                if self.count == 0:
                    print("Edge detection function.")
                mod_frame, self.ROIs = self.ROI_detection(frame.copy())
                # hough_transform()
                self.count += 1
                if self.count % 100 == 0:
                    print("Frame count: {}/{}".format(self.count, self.video_manager.n_frame))
                self.out.write(mod_frame)

                self.paint_retrival(frame)

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

    # ritorna un array di matches (ritornato da BFMatch) e un array di nomi di img per ogni roi
    def paint_retrival(self, frame):
        for roi in self.ROIs:
            # Crop the image
            #print(str(roi[0]) + " " + str(roi[1]) + " " + str(roi[2]) + " " + str(roi[3]))
            crop = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
            crop = cv2.GaussianBlur(crop, (11, 11), 0)

            # kp and des of the crop
            kp_crop, des_crop = find_keypoint(crop)
            th = 60

            kp_crop = cv2.drawKeypoints(crop, kp_crop, color=(0, 255, 0), outImage=None)

            dist = []
            imgs = []
            for n in self.inodes:
                av_dist = matcher(des_crop, self.des_dict[n])
                dist.append(av_dist)
                imgs.append(self.input_img + n)

            s = sorted(zip(dist,imgs))
            imgs = [img for _, img in s]

            i = 0
            for im in imgs:
                i += 1
                if im == self.input_img + "087.png":
                    print("rank: " + str(i))
            print("S: " + str(s))

            d = {}
            d['crop'] = crop
            d['found'] = cv2.imread(imgs[0])
            d['kp_crop'] = kp_crop
            show_frame(d)
