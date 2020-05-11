from source import globals
from source.detection_utils import *
import json
import pandas as pd
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
        self.ROIs_names = []
        self.room = None
        self.data = None # Sarà di tipo pd.DataFrame()

        #Il quadro "x" è nella stanza self.room?
        self.room_dict = {
            #"000.png" : True/False
        }

    def open_video(self, video_name):
        self.cap, self.out = self.video_manager.open_video(video_name, self.input_path, conf['output_path'])

    def close_video(self):
        self.cap.release()
        self.out.release()

    def db_keypoints(self):
        self.inodes = os.listdir(self.input_img)
        self.inodes.remove('.gitkeep')
        self.inodes.sort()

        #Marco, inizializzo room_dict
        for key in self.inodes:
            self.room_dict[key] = False

        print(self.room_dict)

        for inode in self.inodes:
            img = cv2.imread(self.input_img + inode)
            kp, des = find_keypoint(img)

            self.kp_dict[inode] = kp
            self.des_dict[inode] = des

    def set_room_dict(self, img_name, verbose = False):
        """
        Imposta il dizionario interno room_dict a True se la relativa immagine (la chiave del dizionario)
        è contenuta nella stessa stanza dell'immagine 'img_name'.
        False altrimenti.
        La stanza di img_name si ottiene dall'incrocio dei dati del .csv

        :param img_name:
            Nome dell'immagine della quale devo determinare la stanza
        :param verbose:
            Se true, mostra a terminale informazioni aggiuntive sull'andamento del parsing del file csv
        :return:
            No return.
            Imposta solo il dizionario interno.
        """

        # mi serve solo il nome, senza il path
        img_name = os.path.basename(img_name)

        if not self.data:
            self.data = pd.read_csv(conf["data_csv"])
        header = self.data.columns.values

        if verbose:
            print("Detected header: {}".format(header))
            print(self.data)

        curr_row = self.data[self.data["Image"] == img_name]
        self.room = curr_row["Room"].values[0]
        room = self.room

        if verbose:
            print("Il quadro {} è nella stanza {}:\nTitle: {}\nAuthor: {}\nRoom: {}\nImage: {}".format(
                img_name,
                room,
                curr_row[header[0]].values[0],
                curr_row[header[1]].values[0],
                curr_row[header[2]].values[0],
                curr_row[header[3]].values[0]
            ))

        room_paintings = self.data[self.data["Room"] == room]["Image"].values

        # Per sicurezza scorro comunque TUTTO il dizionario,
        # settando a True se il dipinto appartiene alla stanza
        #          a False altrimenti.

        for painting in self.room_dict.keys():
            if painting in room_paintings:
                self.room_dict[painting] = True
            else:
                self.room_dict[painting] = False

        if verbose:
            print("==== room_dict ====\nQuadri nella stanza {}:".format(room))
            for key, value in self.room_dict.items():
                print("{}: {}".format(key, value))
            print("===================")


    def ROI_labeling(self, frame, show_details = False, verbose = False):
        """
        Dato un frame, l'insieme delle ROI al suo interno e i nomi dei file delle immagini relative alle ROI,
        determina il nome del quadro e lo associa alla ROI corrispondente.

        :param frame:
            Il frame da utilizzare
        :return:
            Labeled frame
        """
        labeled_frame = frame.copy()
        if verbose:
            print("ROI LABELING")



        for i, roi in enumerate(self.ROIs):
            row = self.data[self.data["Image"] == self.ROIs_names[i]]
            if row.empty or self.ROIs_names[i] == "":
                text = "Non disponibile nel database [{}]".format(self.ROIs_names[i])
                color = (0,0,255)
            else:
                title = row["Title"].values[0]
                author = row["Author"].values[0]
                text = str(title) + " - " + str(author)
                color = (0,255,0)
            draw_ROI(labeled_frame, roi, text=text, color=color)

            draw_ROI(labeled_frame, (10, frame.shape[0]-100, 0, 0), text="Stanza "+str(self.room), color=(255,255,0), only_text=True)

        if show_details:
            show_frame({"Labeled_frame" : labeled_frame})

        return labeled_frame


    def ROI_detection(self, or_frame):
        gray_frame, marked_frame, ed_frame = edge_detection(or_frame, debug=True,  frame_number=self.count)

        #kp_frame = keypoints_detection(or_frame, show=False)
        roi_frame, ROIs = ccl_detection(or_frame, gray_frame, ed_frame, frame_number=self.count, otsu_opt_enabled=True)
        #ed_frame = cv2.cvtColor(ed_frame, cv2.COLOR_GRAY2BGR)

        return roi_frame, ROIs

    def paint_detection(self, step = 20):
        # Read until video is completed

        start = time.time()

        while (self.cap.isOpened() and self.out.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                if self.count == 0:
                    print("Paint detection function.")
                mod_frame, self.ROIs = self.ROI_detection(frame.copy())

                if self.count % step == 0:# or True:
                    self.retrival_and_rectification(frame.copy(), show_details=False, verbose=False) #show_details = True, verbose = True
                    # ROI LABELING
                    mod_frame = self.ROI_labeling(frame.copy(), show_details=False, verbose=False)



                if self.count % step == 0:
                    print("Frame count: {}/{}".format(self.count, self.video_manager.n_frame))

                if self.count % step == 0:
                    self.out.write(mod_frame)

                self.count += 1

                all_video = True
                if not all_video:
                    break
            else:
                print("Problema durante la lettura del video... stop al frame {}: {}".format(self.count, frame))
                break


        print("Fine paint detection.\nElapsed time: {}".format(time.time() - start))

    def retrival_and_rectification(self, frame, show_details=False, verbose = False):
        if verbose:
            print('Frame {}:'.format(self.count))
        r = 0
        self.ROIs_names = []
        for roi in self.ROIs:
            if verbose:
                print('ROI: {}'.format(r))
            r += 1
            imgs_name = self.paint_retrival(frame, roi, show_details = show_details, verbose=verbose)
            if imgs_name != None:
                av_list = []
                av = 100
                i = 0
                while(av >= 50 and i < 5):
                    av = self.paint_rectification(frame, roi, imgs_name[i], show_details = show_details, verbose=verbose)
                    i += 1
                    av_list.append(av)

                # if(i < len(imgs_name)):
                if(i <= 5):
                    if self.room == None:# and av < 40:
                        self.set_room_dict(img_name=imgs_name[i - 1], verbose=False)

                    self.ROIs_names.append(os.path.basename(imgs_name[i-1]))

                    if show_details:
                        img = cv2.imread(imgs_name[i-1])
                        d = {}
                        d["Chosen Img"] = img
                        show_frame(d)



                else: # Non ha trovato un'immagine corrispondente
                    self.ROIs_names.append("")
            else: # se imgs_name == None
                self.ROIs_names.append("")


    def paint_retrival(self, frame, roi, show_details = False, verbose=False):
        # Crop the image
        crop = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
        # crop = cv2.GaussianBlur(crop, (11, 11), 0)
        if show_details:
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

    def paint_rectification(self, frame, roi, img_name, show_details = False, verbose=False):
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

        if show_details:
            # Show both images
            d = {}
            d["ROI image"] = crop
            d["Rectified image"] = rectified
            d["Retrival image"] = trainImg
            show_frame(d)

        kp_train, des_train = find_keypoint(trainImg)
        kp_rect, des_rect = find_keypoint(rectified)

        av = matcher(des_train, des_rect)
        if verbose:
            print("Img: {}, AV: {}".format(os.path.basename(img_name), av))
        return av
