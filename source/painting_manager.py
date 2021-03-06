from source import globals
from source.detection_utils import *
from yolo import detect, darknet, detect_statues
import json
import pandas as pd
import os

conf = globals.conf
class PaintingManager:
    def __init__(self, video_manager, create_labels):
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
        self.isData = False # Variabile che mi dice se dentro data ho qualcosa
        self.yolo_people_model = None # yolo people model
        self.yolo_statue_model = None # yolo statue model
        self.json_output = {} # the dict that will be parsed to make the ouput_details file
        self.feasible_ROIs = []

        #Il quadro "x" è nella stanza self.room?
        self.room_dict = {
            #"000.png" : True/False
        }

        self.video_name = ""
        self.create_labels = create_labels

    def open_video(self, video_name):
        self.video_name = video_name
        self.cap, self.out = self.video_manager.open_video(video_name, self.input_path, conf['output_path'])

    def close_video(self):
        self.cap.release()
        if not self.create_labels:
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

    def set_room_dict(self, img_name, verbose = False, draw_map=False):
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

        if not self.isData and not self.data:
            self.isData = True
            self.data = pd.read_csv(conf["data_csv"], sep=",")
        header = self.data.columns.values

        if verbose:
            print("Detected header: {}".format(header))
            print(self.data)

        curr_row = self.data[self.data["Image"] == img_name]

        self.room = curr_row["Room"].values[0]
        room = self.room

        if draw_map:
            self.rooms_map_highlight(room=room, color=(0,0,255))

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


    def rooms_map_highlight(self, room=1, color=(255,0,0)):
        """
        Mostra la mappa del museo con evidenziata la stanza in cui ci si trova.

        :return:
        """

        #la riga 0 è vuota. le stanze sono coerenti con i quadri.
        rooms_roi = pd.read_csv("rooms_roi.csv")
        header = rooms_roi.columns.values # room, x, y, w, h

        curr_roi = rooms_roi[rooms_roi["room"] == room]
        map = cv2.imread("."+conf['slash']+"input"+conf['slash']+"map.png")

        cv2.rectangle(map,(curr_roi['x'], curr_roi['y']),(curr_roi['x']+curr_roi['w'], curr_roi['y']+curr_roi['h']), color)

        show_frame({"map": map})
        return map


    def ROI_labeling(self, frame, show_details = False, verbose = False, json_output_details=False):
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

        if not self.isData and not self.data:
            self.isData = True
            self.data = pd.read_csv(conf["data_csv"], sep=",")

        #curr_frame_out = {}
        for i, roi in enumerate(self.ROIs):
            curr_roi_out = {}
            img_name = os.path.basename(self.ROIs_names[i])
            row = self.data[self.data["Image"] == img_name]

            if row.empty:
                if self.ROIs_names[i] == "":
                    roi_id = "ID {} - ".format(i)

                    text = "{}Unidentified object".format(roi_id)
                    if json_output_details:
                        curr_roi_out["text"] = "Unidentified object"
                        curr_roi_out["id"] = i
                    color = (0,255,255) # giallo
                elif self.ROIs_names[i] == "statue":
                    roi_id = "ID {} - ".format(i)

                    text = "{}Statue".format(roi_id)
                    if json_output_details:
                        curr_roi_out["text"] = "Statue"
                        curr_roi_out["id"] = i

                    color = (255, 0, 0)  # blu
            else:
                title = row["Title"].values[0]
                author = row["Author"].values[0]
                painting = "{}".format(img_name)
                roi_id = "ID {} - ".format(i)
                text = str(title) + "\n" + str(author)+ "\n"+ str(roi_id) +str(painting)
                if json_output_details:
                    curr_roi_out["id"] = i
                    curr_roi_out["title"] = title
                    curr_roi_out["author"] = author
                    curr_roi_out["img_name"] = painting
                color = (0,255,0)
            if json_output_details:
                self.json_output["FRAME {}".format(self.count)]["ROI {}".format(i)].update(curr_roi_out)

            draw_ROI(labeled_frame, roi, text=text, color=color)
            draw_ROI(labeled_frame, (10, frame.shape[0]-10, 0, 0), text="Frame {} - Stanza {}".format(self.count, self.room if self.room != None else "non identificata"), color=(255,255,0), only_text=True)

        if show_details:
            show_frame({"Labeled_frame" : labeled_frame})

        return labeled_frame


    def ROI_detection(self, or_frame, otsu_opt_enabled=True):
        gray_frame, marked_frame, ed_frame = edge_detection(or_frame.copy(), debug=True,  frame_number=self.count)
        # kp_frame = keypoints_detection(or_frame, show=False)
        roi_frame, ROIs = ccl_detection(or_frame.copy(),
                                        gray_frame,
                                        ed_frame,
                                        frame_number=self.count,
                                        otsu_opt_enabled=otsu_opt_enabled)
        #roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_GRAY2BGR)

        return roi_frame, ROIs


    def create_labelme_json(self, frame):
        img_name = "{}{}__{}.png".format(conf['labeling_img'], os.path.splitext(self.video_name)[0], self.count)
        cv2.imwrite(img_name, frame)

        json_data = dict({'imageData' : None})
        json_data['version'] = "4.2.10"
        json_data['flags'] = {}
        shapes = []
        for roi in self.ROIs:
            shape = dict({'label': "quadro", 'group_id' : None})
            shape['shape_type'] = "rectangle"
            shape['flags'] = {}
            xy = [(int)(roi[0]), (int)(roi[1])]
            xy2 = [(int)(roi[0] + roi[2]), (int)(roi[1] + roi[3])]
            shape['points'] = [xy, xy2]
            shapes.append(shape)
        json_data['shapes'] = shapes
        json_data['imageHeight'] = frame.shape[0]
        json_data['imagePath'] = "{}{}__{}.png".format(conf['relative_img'], os.path.splitext(self.video_name)[0], self.count)
        json_data['imageWidth'] = frame.shape[1]
        json_name = "{}{}__{}.json".format(conf['labeling_labels'], os.path.splitext(self.video_name)[0], self.count)

        with open(json_name, 'w') as outfile:
            json.dump(json_data, outfile, indent=2)
            print("JSON -> {}\n".format(json_name))
            outfile.close()


    def create_yolomark_txt(self, frame):
        img_name = "{}{}__{}__{}".format(conf['labeling_img'], conf['in_dir'], os.path.splitext(self.video_name)[0], self.count)
        cv2.imwrite("{}.png".format(img_name), frame)

        with open("{}.txt".format(img_name), 'w') as outfile:
            for roi in self.ROIs:
                # values in perc
                cx = (roi[0] + roi[2] / 2) / frame.shape[1]
                cy = (roi[1] + roi[3] / 2) / frame.shape[0]
                w = roi[2] / frame.shape[1]
                h = roi[3] / frame.shape[0]
                row = "0 {} {} {} {}\n".format(cx, cy, w, h)
                outfile.write(row)
            print("TXT -> {}.txt\n".format(img_name))
            outfile.close()


    def init_yolo_people(self):
        """
        Initialize the yolo people detection with config file (conf[yolo people cfg]) and weights file.
        :return:
        """
        self.yolo_people_model = darknet.Darknet(conf['yolo_people_cfg'])
        self.yolo_people_model.load_weights(conf['yolo_people_weights'])


    def init_yolo_statue(self):
        """
        Initialize the yolo statues detection with config file (conf[yolo statue cfg]) and weights file.
        :return:
        """
        self.yolo_statue_model = darknet.Darknet(conf['yolo_statue_cfg'])
        self.yolo_statue_model.load_weights(conf['yolo_statue_weights'])


    def people_detection(self, frame = None, verbose=False, show_details=False):

        if not self.yolo_people_model:
            self.init_yolo_people()
        people_ROIs, people_mod_frame = detect.detect(specific_frame = frame,
                                                      separator=conf['slash'],
                                                      verbose=verbose,
                                                      model=self.yolo_people_model,
                                                      room=self.room) # people detection
        people_mod_frame = people_mod_frame[0]

        if show_details:
            show_frame({"people_mod_frame": people_mod_frame})
        return people_ROIs, people_mod_frame


    def statue_detection(self, frame = None, verbose=False):  # , show_details=False):

        if not self.yolo_statue_model:
            self.init_yolo_statue()
        statue_list = detect_statues.detect_statues(specific_frame = frame, separator=conf['slash'], verbose=verbose, model=self.yolo_statue_model) # statue detection
        # statue_mod_frame = statue_mod_frame[0]

        # if show_details:
        #    show_frame({"people_mod_frame": statue_mod_frame})

        statue_ROIs = []
        for s in statue_list:
            # x, y, w, h
            w = abs(s[1][0] - s[0][0])
            h = abs(s[1][1] - s[0][1])
            roi = list([int(s[0][0]), int(s[0][1]), int(w), int(h)])
            statue_ROIs.append(roi)

        return statue_ROIs


    def clean_ROIs(self, frame, ROIs, ROIs_names, ratio_max=3, statue=False, json_output_details=False):
        temp_ROIs = []
        temp_names = []
        self.feasible_ROIs = []
        # CLEANING ROIs RULES
        # 1. se il rapporto di aspetto è più di 'ratio_max' volte, considero la ROI non ammissibile
        # 2. se l'area è minore di min_area, non è ammissibile
        # 3. se è contenuto in un'altra ROI o contiene un'altra ROI, quello con area minore non è ammissibile (solo se area maggiore è >50% del frame)
        # 4. (opzionale?) se l'overlap supera una certa soglia, la ROI con area minore non è ammissibile
        #ratio_max = 3
        global_area = frame.shape[0] * frame.shape[1]
        min_area = 0.015 * global_area  # almeno una % dell'area totale

        for i, roi in enumerate(ROIs):

            x = roi[0]
            y = roi[1]
            w = roi[2]
            h = roi[3]

            cleaning_boxes = True
            if cleaning_boxes:
                # 1.
                feasible_ratio = abs(w) < abs(ratio_max * (h)) and abs(h) < abs(ratio_max * (w))
                area = w * h
                # 2.
                feasible_area = area > min_area

                if feasible_ratio and feasible_area:
                    temp_ROIs.append([(x, y), (x + w, y + h), w, h])
                    temp_names.append(ROIs_names[i])

        max_overlap = 0.80

        if len(temp_ROIs) == 1:
            return [[temp_ROIs[0][0][0], temp_ROIs[0][0][1], temp_ROIs[0][2], temp_ROIs[0][3]]], temp_names

        # True ROIs array
        trueROIs = []
        trueNames = []

        roi_mask = [True for _ in temp_ROIs]

        for i, roi in enumerate(temp_ROIs):

            # If I discarded this ROI
            if not roi_mask[i]:
                continue

            j = 0
            for roi2 in temp_ROIs:

                if roi[0][0] == roi2[0][0] and roi[0][1] == roi2[0][1] and roi[1][0] == roi2[1][0] and roi[1][1] == \
                        roi2[1][1]:
                    j += 1
                    continue

                width = calculateIntersection(roi[0][0], roi[1][0], roi2[0][0], roi2[1][0])
                height = calculateIntersection(roi[0][1], roi[1][1], roi2[0][1], roi2[1][1])
                area = width * height
                percent = area / (roi[2] * roi[3])

                # must go over the max_overlap AND minor area
                if percent >= max_overlap and roi[2] * roi[3] < roi2[2] * roi2[3]:
                    if statue:
                        roi_mask[i] = False
                    elif temp_names[i] == '':
                        roi_mask[i] = False
                    break

                j += 1


        trueROIs = [[roi[0][0], roi[0][1], roi[2], roi[3]] for i, roi in enumerate(temp_ROIs) if roi_mask[i]]
        trueNames = [e for i, e in enumerate(temp_names) if roi_mask[i]]

        new_json_output = {}

        if not statue:
            self.feasible_ROIs = [index for index, _ in enumerate(temp_ROIs) if roi_mask[index]]
            if json_output_details:
                """
                   removes the ROIs entries in the json 
                   file which are no longer feasible ROIs
                """
                dict_keys = self.json_output["FRAME {}".format(self.count)].keys()
                new_json_output["FRAME {}".format(self.count)] = {}
                counter = 0
                for index, _ in enumerate(temp_ROIs):
                    if index in self.feasible_ROIs:
                        new_json_output["FRAME {}".format(self.count)]["ROI {}".format(counter)] = \
                            self.json_output["FRAME {}".format(self.count)]["ROI {}".format(index)]
                        counter += 1
                self.json_output["FRAME {}".format(self.count)] = new_json_output["FRAME {}".format(self.count)]

        return trueROIs, trueNames


    def paint_detection(self, json_output_details=False, en_segmentation=False, step = 1):
        # Read until video is completed

        start = time.time()

        out_is_ok = False
        if self.create_labels:
            out_is_ok = True
        else:
            out_is_ok = self.out.isOpened()

        self.count = 0
        while (self.cap.isOpened() and out_is_ok):
            ret, frame = self.cap.read()
            if ret == True:
                if self.count == 0:
                    print("START: " + self.video_name)

                if self.create_labels:
                    if self.count % 250 == 0:
                        mod_frame, self.ROIs = self.ROI_detection(frame.copy())
                        self.create_yolomark_txt(frame.copy())
                else:
                    if self.count % step == 0:  # or True:
                        '''
                            'show_details' option set "True" will enable the show_frame(...) functions.
                                           The execution will stop to show each retrieved painting,
                                           its rectification attempt and its rectified version.
                            
                            'verbose'      option set "True" will enable more detailed output.
                                           More info will be printed, including the content of the csv file, 
                                           the matching list of the painting retrieval phase, the scores of the 
                                           painting matching step and all the similarity measures used will be shown. 
                        '''

                        mod_frame, self.ROIs = self.ROI_detection(frame.copy(),
                                                                  otsu_opt_enabled=False)

                        self.retrival_and_rectification(frame.copy(),
                                                        show_details=False,
                                                        verbose=False,
                                                        json_output_details=json_output_details)  # show_details = True, verbose = True
                        statue_ROIs = self.statue_detection(frame.copy(), verbose=False)

                        # Clean all ROIs
                        statue_names = []
                        for _ in statue_ROIs:
                            statue_names.append("statue")
                        statue_ROIs, _ = self.clean_ROIs(frame, statue_ROIs, statue_names, statue=True)

                        all_ROIs = list()
                        all_ROIs.extend(self.ROIs)
                        all_ROIs.extend(statue_ROIs)
                        for _ in statue_ROIs:
                            self.ROIs_names.append("statue")

                        if json_output_details:
                            for index, roi in enumerate(all_ROIs):
                                if self.ROIs_names[index] == "statue":
                                    self.json_output["FRAME {}".format(self.count)].update({"ROI {}".format(index): {}})

                        self.ROIs, self.ROIs_names = self.clean_ROIs(frame, all_ROIs,
                                                                     self.ROIs_names,
                                                                     json_output_details=json_output_details)

                        if en_segmentation:
                            # Segmentation
                            mod_frame = self.segmentation(frame.copy())

                        #show_frame({"ASDAS": mod_frame})
                        # ROI LABELING
                        mod_frame = self.ROI_labeling(mod_frame.copy(),
                                                      show_details=False,
                                                      verbose=False,
                                                      json_output_details=json_output_details)
                        people_ROIs, _ = self.people_detection(frame=mod_frame.copy(),
                                                               show_details=False,
                                                               verbose=False)
                        if json_output_details:
                            for index, roi in enumerate(people_ROIs):
                                self.json_output["FRAME {}".format(self.count)].update({"ROI {}".format(index+len(self.ROIs)): {"text": "Person", "id": index+len(self.ROIs)}})

                        #print(people_ROIs)
                        # show_frame({'Statue Detection', mod_frame})
                        for index, p_roi in enumerate(people_ROIs):
                            color = (255,255,0)#random.choice(colors)
                            c1 = tuple(p_roi[0])
                            c2 = tuple(p_roi[1])

                            room_label = "- Stanza non id." if not self.room else "- Stanza " + str(self.room)
                            label = "ID - {0} {1} {2}".format( index+len(self.ROIs),"Person", room_label)
                            mod_frame = cv2.rectangle(mod_frame, c1, c2, color, 1)

                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                            c2 = list((c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 9))

                            c2 = tuple(np.round(c2).astype(np.int32))
                            c1 = tuple(np.round(c1).astype(np.int32))

                            mod_frame = cv2.rectangle(mod_frame.astype(np.uint8), c1, c2, color, cv2.FILLED)
                            mod_frame = cv2.putText(mod_frame, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                                                    (0,0,0), 1)


                        #show_frame({"ASDASASDSAD2222": mod_frame})

                        if json_output_details:
                            self.json_output["FRAME {}".format(self.count)]["room"] = str(self.room) if self.room != None else "No room"
                            

                self.count += 1

                if self.count % step == 0:
                    print("Frame count: {}/{}".format(self.count, self.video_manager.n_frame))

                #if self.count == 0 or self.count == 400:
                #    show_frame({"SHOW": mod_frame})

                if not self.create_labels and self.count % step == 0:
                    self.out.write(mod_frame)

                all_video = True
                if not all_video:
                    break
            else:
                if json_output_details:

                    print("Building the output_details.json file...")
                    with open("output_details.json", "w") as outfile:
                        json.dump(self.json_output, outfile, indent="   ")

                break

        print("\nElapsed time: {}".format(time.time() - start))
        print("END: " + self.video_name)

    def parse_imgs_details(self, imgs_details):
        curr_d = {}
        for item in imgs_details:
            curr_d["{:10}".format(item[1])] = "{:.3f}".format(item[0])
        return curr_d


    def parse_av(self, av_dict):
        av_d = {}
        for av, img in av_dict.items():
            av_d[img] = av
        return av_d


    def retrival_and_rectification(self, frame, show_details=False, verbose = False, json_output_details=False):

        if verbose:
            print('Frame {}:'.format(self.count))
        r = 0
        self.ROIs_names = []
        curr_json_out = {}

        matching_threshold = 50             # threshold di distanza per selezionare un quadro come corretto
        matching_room_threshold = 35        # "strong threshold" per selezionare il quadro come "ancora" e per scegliere la stanza
        matching_threshold_with_room = 60   # "weak threshold" per selezionare un quadro anche con distanza più alta, se è nella stanza corrente.

        for roi_index, roi in enumerate(self.ROIs):
            comments = ""
            curr_roi_json_out = {}

            if verbose:
                print('ROI: {}'.format(r))
            r += 1
            imgs_name, imgs_details = self.paint_retrival(frame, roi, show_details = show_details, verbose=verbose)

            if json_output_details:
                #making an entry in the output json dictionary
                curr_roi_json_out["imgs_details"] = self.parse_imgs_details(imgs_details)

            if verbose:
                print("==== ROI {} Painting retrieval ====\n{}\n==================================".format(roi_index, "Image name  Match. distance\n" + "\n".join(["{:10}  {:.3f}".format(os.path.basename(x[1]), x[0]) for x in imgs_details])))
            av_dict = {}
            if imgs_name != None:
                template_matchings = []
                av = 100
                i = 0

                while(av >= matching_threshold and i < 5):
                    av = self.paint_rectification(frame, roi, imgs_name[i], show_details = show_details, verbose=verbose)
                    """ PROVA TEMPLATE MATCHING
                    try:
                        curr_roi = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
                        curr_db_paint = cv2.imread(imgs_name[i])
                        curr_roi_area = curr_roi.shape[0] * curr_roi.shape[1]
                        curr_db_paint_area = curr_db_paint.shape[0] * curr_db_paint.shape[1]

                        bigger = curr_roi if curr_roi_area > curr_db_paint_area else curr_db_paint
                        smaller = curr_db_paint if curr_roi_area > curr_db_paint_area else curr_roi

                        temp_match_measure = cv2.matchTemplate(bigger, smaller, cv2.TM_SQDIFF_NORMED)
                        print(temp_match_measure)
                    except:
                        pass
                        #print("Qualche errore nella template matching.")
                    """

                    av_dict[av] = imgs_name[i]
                    i += 1

                # if(i < len(imgs_name)):
                # prima era i < 5
                if(i < 5 and self.room == None):
                    if show_details:
                        img = cv2.imread(imgs_name[i - 1])
                        d = {}
                        d["Chosen Img"] = img
                        show_frame(d)

                    comments += "Matching found (av < {}): chosen image is {}.\n".format(matching_threshold, os.path.basename(imgs_name[i-1]))
                    if av < matching_room_threshold:
                        self.set_room_dict(imgs_name[i-1], verbose=verbose, draw_map=True)
                        comments+="Strong matching (av <  {}): the room is set to the matched painting's room.\n".format(matching_room_threshold)

                    self.ROIs_names.append(imgs_name[i-1])

                elif(i < 5 and self.room != None):
                    # come sopra ma controllo anche che il quadro sia nella stanza
                    img_name = os.path.basename(imgs_name[i-1])
                    if self.room_dict[img_name] == True:
                        self.ROIs_names.append(imgs_name[i-1])
                        comments += "Matching found (av < {}) and it's in the same room: chosen image is {}.\n".format(matching_threshold,
                                                                                               os.path.basename(
                                                                                                   imgs_name[i-1]))
                    else:
                        # Per il momento stringa vuota
                        self.ROIs_names.append("")
                        comments += "Matching found (av < {}) but paiting is not in the same room: NO MATCH.\n".format(matching_threshold)

                elif(self.room != None):
                    av_keys = (list(av_dict.keys()))
                    av_keys.sort()
                    found = False
                    for k in av_keys:
                        if k < matching_threshold_with_room and not found: # and checkimage in stanza con il dict
                            img_name = os.path.basename(av_dict[k])
                            # allora hai trovato l'immagine
                            if self.room_dict[img_name] == True:
                                self.ROIs_names.append(av_dict[k])
                                comments += "No match found (av > {}) but this paiting is in the same room, and therefore its distance is feasible (av < {}): chosen image is {}.\n".format(
                                    matching_threshold, matching_threshold_with_room,img_name)
                                found = True

                    if not found:
                        self.ROIs_names.append("")

                        comments += "No feasible match found.\n"
                else:
                    self.ROIs_names.append("")
                    comments += "No feasible match found.\n"
            else:  # se imgs_name == None
                self.ROIs_names.append("")
                comments += "No feasible match found.\n"

            if json_output_details:
                curr_roi_json_out["av_distances"] = self.parse_av(av_dict)
                curr_roi_json_out["comments"] = list(filter(None,comments.split("\n")))
                curr_json_out["ROI {}".format(roi_index)] = curr_roi_json_out
            if verbose:
                print("===== COMMENTS: FRAME {} ROI {} =====\n{}======== END COMMENTS ========".format(self.count, roi_index, comments))

        if json_output_details:
            #making the partial entry for the current frame, it will also be updated by the ROI labeling function
            self.json_output["FRAME {}".format(self.count)] = curr_json_out


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
            return None, [(0, "No match")]

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
        return imgs, s

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

        if show_details:
            show_frame({"Matches": matchImg})

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

    def segmentation(self, frame):
        ROIs = [roi for i, roi in enumerate(self.ROIs) if self.ROIs_names[i] != ""]
        names = [names for i, names in enumerate(self.ROIs_names) if self.ROIs_names[i] != ""]

        for i, roi in enumerate(ROIs):
            #show_frame({"Frame":frame})

            x = roi[0]
            y = roi[1]
            w = roi[2]
            h = roi[3]
            Lx = int(round(w * 30/100))
            Ly = int(round(h * 30/100))

            #Lx_2 = int(round(w * 5/100))
            #Ly_2 = int(round(h * 5/100))

            y1 = y-Ly if y-Ly >= 0 else 0
            y2 = y+h+Ly if y+h+Ly <= frame.shape[0] else frame.shape[0]
            x1 = x-Lx if x-Lx >= 0 else 0
            x2 = x+w+Lx if x+w+Lx <= frame.shape[1] else frame.shape[1]

            img = frame[y1:y2,x1:x2]
            #show_frame({"IMG":img})
            img_rect = img.copy()
            Nx = abs(x1 - x)
            Ny = abs(y1 - y)

            mask = np.zeros(img.shape[:2],np.uint8)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            """
            if names[i] == "statue":
                ShiftX = Nx - Lx_2 if Nx - Lx_2 >= 0 else 0
                ShiftY = Ny - Ly_2 if Ny - Ly_2 >= 0 else 0
                ShiftW = w + Lx_2 if w + Lx_2 <= img.shape[1] else img.shape[1]
                ShiftH = h + Ly_2 if h + Ly_2 <= img.shape[0] else img.shape[0]
                rect = (ShiftX, ShiftY, ShiftW, ShiftH)
            else:
            """
            rect = (Nx, Ny, w, h)
            cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            img_2 = img*mask2[:,:,np.newaxis]

            img[mask2 == 1] = np.clip(img[mask2 == 1].astype(np.int32) + np.array((0, 50, 0), dtype=np.uint8), 0, 255).astype(np.uint8)
            #img.clip(0,255).astype(np.uint8)
            #frame[Ny:Ny+h,Nx:Nx+w] = img
            # show_frame({"OUT":img})
        return frame

