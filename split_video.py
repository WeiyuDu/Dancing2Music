import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      print ('Read a new frame: ', success)
      cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", default = "/mnt/volume2/Data/weiyu/multiperson/dance1.mp4", help="path to video")
    a.add_argument("--pathOut", default = "/mnt/volume2/Data/weiyu/multiperson/dance1_frames", help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)