from source import edge_detection, hough_transform, globals

conf = globals.conf

if __name__ == '__main__':
    input_path = conf['input_path'] + '000' + conf['slash']
    video_name = "VIRB0401"
    video_name = "VIRB0391"

    manager = globals.VideoManager()
    cap, out = manager.open_video(video_name, input_path, conf['output_path'])

    # Read until video is completed
    count = 0
    while (cap.isOpened() and out.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            if count == 0:
                print("Edge detection function.")

            mod_frame = edge_detection(frame, debug=True, corners=False, frame_number = count)

            # hough_transform()
            count += 1
            if count % 100 == 0:
                print("Frame count: {}/{}".format(count, manager.n_frame))
            out.write(mod_frame)

            all_video = True
            if not all_video:
                break
        else:
            break

    print("Fine edge_detection.")

    cap.release()
    out.release()

