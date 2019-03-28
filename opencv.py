import cv2
import imutils
import numpy as np

# global variables
bg = None
bg2= None
flag = None

#-------------------------------------------------------------------------------
# Function - To find the running average over the background
#-------------------------------------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def run_avg2(image, aWeight):
    global bg2
    # initialize the background
    if bg2 is None:
        bg2 = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg2, aWeight)

#-------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
#-------------------------------------------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)




#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    
    area=0
    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]
        roi2 = frame[top:bottom, right-250:left-250]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)
            #hand2 = segment(gray2)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                area = cv2.contourArea(segmented)
                cv2.imshow("Thesholded", thresholded)
                '''
                if ( (area > 10000) and flag ):
                    flag=False
                    print('zoom in')
                elif ( (area < 10000) and not(flag) ):
                    flag=True
                    print('zoom out')
                cv2.imshow("Thesholded2", thresholded)
                '''
                
        if num_frames < 30:
            run_avg2(gray2, aWeight)
        else:
            # segment the hand region
            #hand = segment(gray)
            hand2 = segment(gray2)

            # check whether hand region is segmented
        
            if hand2 is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand2

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right-250, top)], -1, (0, 0, 255))
                area = area + cv2.contourArea(segmented)
                area=area/2
                #print(area)
                if ( (area > 10000) and flag ):
                    flag=False
                    print('zoom in')
                elif ( (area < 10000) and not(flag) ):
                    flag=True
                    print('zoom out')
                cv2.imshow("Thesholded2", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        cv2.rectangle(clone, (left-250, top), (right-250, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()
