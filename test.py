import cv2
import time
import pynput

from invoke import run

#cmd = "xdotool search --onlyvisible --class 'Chrome' windowfocus key space"
from pynput.keyboard import Key, Controller
keyboard=Controller()
#keyboard.press('a')



cap = cv2.VideoCapture(0)

r_t = (70,200)
r_b = (200,370)

last_jump = time.time()

while (True):
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)
	frame_copy = frame.copy()

	frame =  cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	cv2.rectangle(frame_copy, r_t,r_b, (0,255,0), 2)
	cv2.imshow("Original_With_ROI", frame_copy)

	roi  = frame[100:500,100:500]


	ret,thresh1 = cv2.threshold(roi,100,255,cv2.THRESH_BINARY)


	thresh1 = 255 - thresh1;

	contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame_copy, contours, -1, (0,255,0), 3)


	if len(contours) != 0:
		x,y,w,h = 0,0,0,0
		c = max(contours, key = cv2.contourArea)
		x,y,w,h = cv2.boundingRect(c)

		diff = time.time() - last_jump

		if(h*w  < 12000)   & (diff > 0.5)  :
                        print("jump",diff)
                        keyboard.press(Key.space)
                        keyboard.release(Key.space)
                        last_jump = time.time()


	k = cv2.waitKey(33)
	if k == 27:
		cv2.destroyAllWindows()
		break
