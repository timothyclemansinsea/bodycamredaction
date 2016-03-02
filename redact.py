import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import numpy
import string, random
import os
import SkinDetector
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to the video file")
ap.add_argument("-o", "--output", help="path to the video file")
args = vars(ap.parse_args())
print 'setting up camera'
#camera = cv2.VideoCapture(args["input"])

print 'making frames folder'
#frames_folder = 'working_space'
frames_folder = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
os.system('mkdir %s/' % (frames_folder))
##os.system('ffmpeg -i %s -r 1 -f image2 %s/%%08d.jpg' % (args['input'], frames_folder))
os.system('ffmpeg -i %s -f image2 %s/%%08d.jpg' % (args['input'], frames_folder))
os.system('ffmpeg -i %s -f image2 %s/%%08d.jpg' % (args['input'], frames_folder))
grays = []
# initialize the first frame in the video stream
firstFrame = None
avg = None
i = 0
for f in sorted(os.listdir(frames_folder)):
    #print i
    i += 1
    # grab the current frame and initialize the occupied/unoccupied
    # text
    #(grabbed, frame) = camera.read()
    frame = cv2.imread(frames_folder+'/'+f)
    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    converted[:,:,0] = cv2.equalizeHist(converted[:,:,0])
    
    converted = cv2.cvtColor(converted, cv2.COLOR_YCR_CB2BGR)
    converted = cv2.cvtColor(converted, cv2.COLOR_BGR2HSV)
    #skinMask = SkinDetector.process(frame)
    lower_thresh = numpy.array([0,5,5], dtype=numpy.uint8)
    upper_thresh = numpy.array([180,255,255], dtype=numpy.uint8)
    #lower_thresh = numpy.array([0, 50, 0], dtype=numpy.uint8)
    #upper_thresh = numpy.array([120, 150, 255], dtype=numpy.uint8)
    skinMask = cv2.inRange(converted, lower_thresh, upper_thresh)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    (cnts, _) = cv2.findContours(skinMask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if len(cnts) > 0:
        print 'HAS Contours', i, len(cnts)
    temp = np.zeros(frame.shape,np.uint8)
    #d = cv2.drawContours(temp,cnts,0,255,-1)
    #areas = []
    for c in cnts:
        #(x, y, w, h) = cv2.boundingRect(c)
        #areas.append((w*h, w, h))
        #detections.append(((x, y, w, h), i))
        #temp = np.zeros(frame.shape,np.uint8)
        d = cv2.drawContours(temp,[c],0,255,-1)
        x = np.where(temp != 0)
        frame[x[:2]] = blurred[x[:2]]
    filename = '%08d.jpg' % (i)
    cv2.imwrite('%s/%s' % (frames_folder, filename), frame)
ffmpeg_cmd = '%sffmpeg -i %s/%%08d.jpg -y -r 24 -vcodec libx264 -crf 22 -preset ultrafast -b:a 32k -strict -2 %s' % ('./', frames_folder, args["output"])
print ffmpeg_cmd
os.system(ffmpeg_cmd)
#os.system('rm -rf %s' % (frames_folder))
