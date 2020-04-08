import cv2
import imutils
import numpy as np
import pytesseract
from time import sleep
from loguru import logger
import matplotlib.pyplot as plt


def mask_color(image):
	lower = np.array([50, 50, 50][::-1], dtype=np.uint8)
	upper = np.array([100, 255, 100][::-1], dtype=np.uint8)

	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask=mask)

	return output

def roi(image):
	
	top_left = (10, 445)
	top_right = (250, 435)
	bottom_right = (265, 455)
	bottom_left = (20, 465)

	pts = np.array([[top_left, top_right, bottom_right, bottom_left]])
	mask = np.zeros(image.shape, dtype=np.uint8)

	channel = image.shape[2]
	ignore_mask_color = (255,)*channel

	cv2.fillPoly(mask, pts, ignore_mask_color)

	masked_image = cv2.bitwise_and(image, mask)

	# now crop

	rect = cv2.boundingRect(pts)
	x,y,w,h = rect
	cropped = image[y:y+h, x:x+w].copy()

	return cropped, masked_image

def main():
	video = cv2.VideoCapture('output.mp4')

	if not video.isOpened():
		logger.error('Error opening file.')
		raise Exception('FileOpenError')

	frames = 0
	while frames <= 300:
		frames += 1
		ret, img = video.read()

		resized_image = imutils.resize(img, width=1024) # coz costly to work on full resolution
	
		cropped, masked = roi(resized_image)
		rotated = imutils.rotate(cropped, -2.386) # straighten the image

		zoomed = imutils.resize(rotated, width=1000)

		# masked = mask_color(zoomed)
		gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)

		# threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		ret, threshold = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV) #barely works
		img_rgb = cv2.cvtColor(threshold, cv2.COLOR_BGR2RGB)
		# img_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
		string = pytesseract.image_to_string(img_rgb)

		cv2.imshow('window', img_rgb)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

		logger.debug(f'frame: {frames}: {string}')

		# break

	logger.debug('finished processing')
	video.release()
	cv2.destroyAllWindows()

	del video



if __name__ == '__main__':
	main()