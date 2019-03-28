import cv2
import numpy as np;

# Read image
cap=cv2.VideoCapture(0)

while True:
    (grabbed, frame) = cap.read()
    #im = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(im)

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)

    if (cv2.waitKey(25) & 0xFF == ord('q')):
      break
