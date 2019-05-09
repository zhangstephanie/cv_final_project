import sys
import cv2

try:
    i = 1
    name = "./data/images/bringback_steph_1_{:0>4d}.jpg".format(i)
    imageToProcess = cv2.imread(name)
    print(imageToProcess)

    for
        i += 1
        imageToProcess = cv2.imread("./data/images/bringback_steph_1_{:0>4d}.jpg".format(i))
        if imageToProcess is None:
            break
        else:
            print(i)
            resizeImage = cv2.resize(imageToProcess, (0,0), fx=0.3, fy=0.3)
            cropImage = halfImage[75:,:]
            cv2.imwrite("./data/croprescale/cr_bringback_steph_1_{:0>4d}.jpg".format(i),cropImage)
            cv2.imshow("small image",cropImage)
            cv2.waitKey(0)

except Exception as e:
    print(e)
    sys.exit(-1)
