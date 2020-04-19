from source import edge_detection, hough_transform, globals

conf = globals.conf

if __name__ == '__main__':
    input_path = conf['input_path'] + '000' + conf['slash']
    video_name = "VIRB0401"

    manager = globals.VideoManager()
    cap, out = manager.open_video(video_name, input_path, conf['output_path'])

    # Read until video is completed
    while (cap.isOpened() and out.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            mod_frame = edge_detection(frame, debug=True, corners=False)

            # hough_transform()

            out.write(mod_frame)

        else:
            break

    cap.release()
    out.release()

