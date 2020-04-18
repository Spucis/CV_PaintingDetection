from source import edge_detection, hough_transform
from json import *
from io import *

file = open('.\\conf.json', 'r').read()
conf = JSONDecoder().decode(s=file)
print("Config file: \n{}".format(conf))

edge_detection(debug = False, force_all_video=True, corners=False, conf = conf)
hough_transform()

