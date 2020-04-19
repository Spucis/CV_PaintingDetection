from json import *

file = open('conf.json', 'r').read()
conf = JSONDecoder().decode(s=file)
print("Config file: \n{}".format(conf))
