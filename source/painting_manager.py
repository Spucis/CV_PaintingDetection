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
                mod_frame, self.ROIs = self.ROI_detection(frame.copy())
                # hough_transform()

                # Paint retrival e rectification ogni 50 frame
                if self.count % 50 == 0:
                    for roi in self.ROIs:
                        img_name = self.paint_retrival(frame, roi)
                        if img_name != -1:
                            self.paint_rectification(frame, roi, img_name)

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

    # ritorna il nome img corrispondente alla Roi
    def paint_retrival(self, frame, roi):
        # Crop the image
        #print(str(roi[0]) + " " + str(roi[1]) + " " + str(roi[2]) + " " + str(roi[3]))
        crop = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
        #crop = cv2.GaussianBlur(crop, (11, 11), 0)

        # kp and des of the crop
        kp_crop, des_crop = find_keypoint(crop)
        if kp_crop is None or des_crop is None:
            return -1

        kp_crop = cv2.drawKeypoints(crop, kp_crop, color=(0, 255, 0), outImage=None)

        dist = []
        imgs = []
        for n in self.inodes:
            av_dist = matcher(des_crop, self.des_dict[n])
            if av_dist < 0:
                return -1

            dist.append(av_dist)
            imgs.append(self.input_img + n)

        if len(dist) == 0:
            return -1

        s = sorted(zip(dist,imgs))
        imgs = [img for _, img in s]

        i = 0
        print('Frame {}:'.format(self.count))
        for d, im in s:
            print('{} : d = {};\timg = {}'.format(i, d, im))
            i += 1
        print('\n\n\n')

        d = {}
        d['found'] = cv2.imread(imgs[0])
        d['kp_crop'] = kp_crop
        show_frame(d)

        return imgs[0]

    # prendo la roi e il nome img corrispondente e mostro l'img raddrizzata
    def paint_rectification(self, frame, roi, img_name):
        crop = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        # cv2.imwrite("./crop.png", crop)
        #show_frame({"Crop": crop})

        if img_name == '':
            return

        trainImg = cv2.imread(img_name)

        # KEYPOINTS METHOD

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # quadro centrale
        d1 = self.des_dict[os.path.basename(img_name)]
        k1 = self.kp_dict[os.path.basename(img_name)]

        k2, d2 = find_keypoint(crop)
        if k2 is None or d2 is None:
            return

        kp_crop = cv2.drawKeypoints(crop, k2, None, color=(0, 255, 0), flags=0)
        # show_frame({"Crop Keypoints": kp_crop})

        # Match descriptors per gli indici dei k1 e k2
        matchList = bf.match(d1, d2)

        if len(matchList) < globals.match_th:
            return

        # take only matches with distance less than a threshold
        # trueMatches = [m for m in matchList if m.distance < 50]

        # Sort matches based on distances
        # sortMatches = sorted(matchList, key=lambda val: val.distance)

        matchImg = cv2.drawMatches(trainImg, k1, crop, k2, matchList, flags=2, outImg=None)
        show_frame({"Matches": matchImg})

        # Coordinate dei keypoints
        trainPoints = []
        queryPoints = []
        for m in matchList:
            i = m.trainIdx
            j = m.queryIdx
            trainPoints.append(k2[i].pt)
            queryPoints.append(k1[j].pt)

        """
        # HOUGH LINES METHOD

        # Tentativo per trovare i corner del quadro
        # Threshold rumorose
        th = 400
        TH = 800
        mod_f = cv2.Canny(crop, th, TH, apertureSize=5, L2gradient=True)
        show_frame({"Canny Crop": mod_f})
        lines = cv2.HoughLines(mod_f, 1, np.pi/180, 150)
        for l in lines:
            for r, theta in l:
                # Stores the value of cos(theta) in a
                a = np.cos(theta)
                # Stores the value of sin(theta) in b
                b = np.sin(theta)
                # x0 stores the value rcos(theta)
                x0 = a * r
                # y0 stores the value rsin(theta)
                y0 = b * r
                # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                x1 = int(x0 + 1000 * (-b))
                # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                y1 = int(y0 + 1000 * (a))
                # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                x2 = int(x0 - 1000 * (-b))
                # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
                y2 = int(y0 - 1000 * (a))
                lines_f = cv2.line(crop, (x1, y1), (x2, y2), (0, 0, 255), 2)

        show_frame({"Crop Lines": lines_f})
        """

        """
        # CORNERS METHOD
        th = 0.05
        # crop corners
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerHarris(gray_crop, 2, 3, 0.04)
        # Apply threshold for corners
        img = crop.copy()
        img[corners > th * corners.max()] = [0, 0, 255]
        show_frame({"Crop Corners": img})
        # Find coordinates of corners
        trainPoints = crop[corners > th * corners.max()]

        # train img corners
        gray_train = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerHarris(gray_train, 2, 3, 0.04)
        # Apply threshold for corners
        img = trainImg.copy()
        img[corners > th * corners.max()] = [0, 0, 255]
        show_frame({"Train img Corners": img})
        # Find coordinates of corners
        queryPoints = trainImg[corners > th * corners.max()]
        """

        # Finds transformation matrix
        trainPoints = np.array(trainPoints, dtype=np.float32)
        queryPoints = np.array(queryPoints, dtype=np.float32)
        H, retval = cv2.findHomography(trainPoints, queryPoints, cv2.RANSAC)

        # Rectify crop image
        rectified = cv2.warpPerspective(crop, H, (trainImg.shape[1], trainImg.shape[0]))
        # show_frame({"Rectified image": rectified})

        # Show both images
        cv2.imshow("ROI image", crop)
        cv2.imshow("Rectified image", rectified)
        cv2.imshow("Retrival image", trainImg)
        cv2.waitKey()
        cv2.destroyAllWindows()
