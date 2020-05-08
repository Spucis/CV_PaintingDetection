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

    def db_keypoints(self):
        self.inodes = os.listdir(self.input_img)
        self.inodes.remove('.gitkeep')
        self.inodes.sort()

        for inode in self.inodes:
            img = cv2.imread(self.input_img + inode)
            kp, des = find_keypoint(img)

            self.kp_dict[inode] = kp
            self.des_dict[inode] = des

    def ROI_detection(self, or_frame):
        gray_frame, marked_frame, ed_frame = edge_detection(or_frame, debug=True,  frame_number=self.count)
        # kp_frame = keypoints_detection(or_frame, show=False)
        roi_frame, ROIs = ccl_detection(or_frame, gray_frame, ed_frame, frame_number=self.count)
        # ed_frame = cv2.cvtColor(ed_frame, cv2.COLOR_GRAY2BGR)
        return roi_frame, ROIs

    def paint_detection(self):
        # Read until video is completed
        while (self.cap.isOpened() and self.out.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                if self.count == 0:
                    print("Edge detection function.")
                mod_frame, self.ROIs = self.ROI_detection(frame.copy())
                if self.count % 50 == 0:
                    self.retrival_and_rectification(frame.copy())
                self.count += 1
                # ??????if self.count % 100 == 0:
                # print("Frame count: {}/{}".format(self.count, self.video_manager.n_frame))
                self.out.write(mod_frame)
                all_video = True
                if not all_video:
                    break
            else:
                print("Problema durante la lettura del video... stop al frame {}: {}".format(self.count, frame))
                break
        print("Fine edge_detection.")

    def retrival_and_rectification(self, frame):
        print('Frame {}:'.format(self.count))
        r = 0
        for roi in self.ROIs:
            print('ROI: {}'.format(r))
            r += 1
            imgs_name = self.paint_retrival(frame, roi)
            if imgs_name != None:
                av_list = []
                av = 100
                i = 0
                while(av >= 50 and i < 5):
                    av = self.paint_rectification(frame, roi, imgs_name[i])
                    i += 1
                    av_list.append(av)

                # if(i < len(imgs_name)):
                if(i < 5):
                    img = cv2.imread(imgs_name[i-1])
                    d = {}
                    d["Chosen Img"] = img
                    show_frame(d)

    def paint_retrival(self, frame, roi):
        # Crop the image
        crop = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
        # crop = cv2.GaussianBlur(crop, (11, 11), 0)
        d = {}
        d['CROP'] = crop
        show_frame(d)

        # kp and des of the crop
        kp_crop, des_crop = find_keypoint(crop)
        if kp_crop is None or des_crop is None:
            return None

        kp_crop = cv2.drawKeypoints(crop, kp_crop, color=(0, 255, 0), outImage=None)
        dist = []
        imgs = []
        for n in self.inodes:
            av_dist = matcher(des_crop, self.des_dict[n])
            dist.append(av_dist)
            imgs.append(self.input_img + n)

        s = sorted(zip(dist,imgs))
        imgs = [img for _, img in s]

        # d = {}
        # d['found'] = cv2.imread(imgs[0])
        #
        # d['kp_crop'] = kp_crop
        # show_frame(d)
        return imgs

    def paint_rectification(self, frame, roi, img_name):
        crop = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        if img_name == '':
            return 100
        trainImg = cv2.imread(img_name)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        kp_train = self.kp_dict[os.path.basename(img_name)]
        des_train = self.des_dict[os.path.basename(img_name)]

        kp_crop, des_crop = find_keypoint(crop)
        if kp_crop is None or des_crop is None:
            return 100

        # kp_crop = cv2.drawKeypoints(crop, k2, None, color=(0, 255, 0), flags=0)
        # show_frame({"Crop Keypoints": kp_crop})

        # Match descriptors per gli indici dei k1 e k2
        matchList = bf.match(des_train, des_crop)

        if len(matchList) < globals.match_th:
            return 100

        # Sort matches based on distances
        # sortMatches = sorted(matchList, key=lambda val: val.distance)
        matchImg = cv2.drawMatches(trainImg, kp_train, crop, kp_crop, matchList, flags=2, outImg=None)
        #show_frame({"Matches": matchImg})

        # Coordinate dei keypoints
        trainPoints = []
        queryPoints = []
        for m in matchList:
            i = m.trainIdx
            j = m.queryIdx
            trainPoints.append(kp_crop[i].pt)
            queryPoints.append(kp_train[j].pt)

        # Finds transformation matrix
        trainPoints = np.array(trainPoints, dtype=np.float32)
        queryPoints = np.array(queryPoints, dtype=np.float32)
        H, retval = cv2.findHomography(trainPoints, queryPoints, cv2.RANSAC)
        # Secondo me si può dire che se H ritornato = None, allora è un muro

        try:
            # Rectify crop image
            rectified = cv2.warpPerspective(crop, H, (trainImg.shape[1], trainImg.shape[0]))
            # show_frame({"Rectified image": rectified})
        except:
            return 100

        # Show both images
        d = {}
        d["ROI image"] = crop
        d["Rectified image"] = rectified
        d["Retrival image"] = trainImg
        show_frame(d)

        kp_train, des_train = find_keypoint(trainImg)
        kp_rect, des_rect = find_keypoint(rectified)

        av = matcher(des_train, des_rect)
        print("Img: {}, AV: {}".format(os.path.basename(img_name), av))
        return av
