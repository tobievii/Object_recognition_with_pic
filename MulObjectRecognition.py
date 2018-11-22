import cv2
import numpy as np
import pyglet
import time

MIN_MATCH_COUNT = 30
sift = cv2.xfeatures2d.SIFT_create()

# Feature matching with image and video capture
FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm = FLANN_INDEX_KDITREE, tree = 5)
flann = cv2.FlannBasedMatcher(flannParam, {})

# Image 1
trainImg = cv2.imread("models/malkisy.jpg", 0)
trainKP, trainDesc = sift.detectAndCompute(trainImg, None)

# Image 2
trainImg2 = cv2.imread("models/tilly.jpg", 0)
trainKP2, trainDesc2 = sift.detectAndCompute(trainImg2, None)

cam = cv2.VideoCapture(0)
need_music1 = True
need_music2 = True
while True:
    ret, QueryImgBGR = cam.read()
    QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
    queryKP, queryDesc = sift.detectAndCompute(QueryImg, None)

    #Matching 1
    matches = flann.knnMatch(queryDesc, trainDesc, k = 2)
    goodMatch = []
    for m, n in matches:
        if(m.distance<0.7*n.distance):
            goodMatch.append(m)

    #Matching 2
    matches2 = flann.knnMatch(queryDesc, trainDesc2, k = 2)
    goodMatch2 = []
    for m, n in matches2:
        if(m.distance<0.7*n.distance):
            goodMatch2.append(m)

    # Homography
    if(len(goodMatch)>MIN_MATCH_COUNT or len(goodMatch2)>MIN_MATCH_COUNT):
        tp = []
        qp = []
        tp2 = []
        qp2 = []
        if(len(goodMatch)>MIN_MATCH_COUNT):
            #Draw border 1
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp, qp = np.float32((tp, qp))
            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
            h, w = trainImg.shape
            trainBorder = np.float32([[[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]])
            queryBorder = cv2.perspectiveTransform(trainBorder, H)
            cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (0, 255, 0), 5)
            if(need_music1 == True):
                music = pyglet.resource.media('sounds/malkist.mp3') #ดึงไฟล์เสียงเข้ามา
                music.play()
                need_music1 = False

        if(len(goodMatch2)>MIN_MATCH_COUNT):
            #Draw border 2
            for m in goodMatch2:
                tp2.append(trainKP2[m.trainIdx].pt)
                qp2.append(queryKP[m.queryIdx].pt)
            tp2, qp2 = np.float32((tp2, qp2))
            H2, status2 = cv2.findHomography(tp2, qp2, cv2.RANSAC, 3.0)
            h2, w2 = trainImg2.shape
            trainBorder2 = np.float32([[[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]])
            queryBorder2 = cv2.perspectiveTransform(trainBorder2, H2)
            cv2.polylines(QueryImgBGR, [np.int32(queryBorder2)], True, (255, 255, 0), 5)
            if(need_music2 == True):
                music = pyglet.resource.media('sounds/tilly.mp3') #ดึงไฟล์เสียงเข้ามา
                music.play()
                need_music2 = False
    else:
        print("Not Enough match found- %d/%d"%(len(goodMatch), MIN_MATCH_COUNT))
        need_music1 = True
        need_music2 = True
    cv2.imshow('result', QueryImgBGR)
    if cv2.waitKey(10) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()