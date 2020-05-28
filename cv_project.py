from source import edge_detection, hough_transform, globals
from source.painting_manager import *
import split_dataset

conf = globals.conf

#video_name = "VIRB0401" #000
video_name = "VIRB0391.mp4"

video_name = "VIRB0400.mp4"
#video_name = "VIRB0402"

video_name = "20180206_114720.mp4" # 002 # IMP 1
#video_name = "20180206_113716.mp4"
#video_name = "20180206_112930.mp4"
#video_name = "20180206_114408.mp4"
#video_name = "20180206_114009.mp4"

#video_name = "GOPR1926" #003
#video_name = "GOPR1932.MP4"

#video_name = "IMG_3814.MOV" #004
#video_name = "IMG_3812.MOV"

#video_name = "VIRB0420.mp4" #008
#video_name = "VIRB0421" #008
#video_name = "VIRB0413.mp4" #008


#video_name = "VID_20180529_112706.mp4" # 010 # IMP 2 Primo da fare vedere
#video_name = "VID_20180529_112951.mp4" # 010

#video_name = "VID_20180529_112849.mp4" # 010
#video_name = "VID_20180529_112828" # 010

# Labels
create_labels = False

p_manager = PaintingManager(globals.VideoManager(create_labels), create_labels)

print("CREATE LABELS: {}".format(create_labels))

if create_labels:

    split_dataset.split_dataset(conf)

    nodes = os.listdir("{}{}".format(conf['input_path'], conf['in_dir']))
    nodes.sort()
    for node in nodes:
        p_manager.open_video(node)
        p_manager.paint_detection()
        p_manager.close_video()
else:
    p_manager.db_keypoints()
    p_manager.open_video(video_name)
    # 'json_ouput_details' will produce a json file "output_details.json"
    # containing all the details made by this program
    # 'step': the frame will be processed each 'step' frames
    p_manager.paint_detection(json_output_details=True, step=20)
    p_manager.close_video()
