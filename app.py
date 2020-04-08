import os
import sys
import cv2
import click
import imutils
import numpy as np
import pytesseract
from time import sleep
from copy import deepcopy
from loguru import logger
from fuzzywuzzy import fuzz
from AutoQueue import AutoQueue
import matplotlib.pyplot as plt

logger.add(sys.stderr, level='WARNING')
logger.remove(0) # workaround for log level

GLOBAL_BUFFER = AutoQueue(7*60)
CLIP_COUNTER = 0

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

def straighten(image):
	return imutils.rotate(image, -2.386)


def ocr(image):
	text = pytesseract.image_to_string(image)
	return text

def parse_text(text, ign='Rider'):
	pred = fuzz.partial_ratio(text, ign)
	if len(text) > len(ign):
		if pred > 80:
			logger.debug('text')
			return True
	return False

def check_kill(image, ign):
	text = ocr(image)
	return parse_text(text, ign)

def make_clip():
	global CLIP_COUNTER
	global GLOBAL_BUFFER
	CLIP_COUNTER += 1
	clip_name = os.path.join('clips', f'clip_{CLIP_COUNTER}.avi')
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	clip = cv2.VideoWriter(clip_name, fourcc, 60, (1920, 1080))

	BUFFER_COPY = AutoQueue()
	BUFFER_COPY.queue = deepcopy(GLOBAL_BUFFER.queue)
	while not BUFFER_COPY.empty():
		frame = BUFFER_COPY.get(block=False)
		clip.write(frame)

	click.secho(f'clip saved at {clip_name}', fg='green')

@click.command()
@click.argument('filename', type=click.Path(exists=True), default=None)
@click.argument('ign', type=str)
@click.option('--log', '-l', type=click.Choice(['DEBUG', 'INFO', 'WARNING']), default='WARNING')
@logger.catch
def main(filename, ign, log):
	try:
		click.secho('Ctrl+C to abort.', fg='yellow')
		# click.echo('press q')
		video = cv2.VideoCapture(filename)

		if not video.isOpened():
			logger.error('Error opening file.')
			raise Exception('FileOpenError')


		if log:
			logger.add(sys.stderr, level=log)
			logger.remove(1)

		frames = 0
		KILL_CHECK = False
		KILL_FRAME = 0
		CHECK = True
		# while frames <= 300:
		while True:
			frames += 1
			ret, img = video.read()
			if ret:
				GLOBAL_BUFFER.put(img)
				resized_image = imutils.resize(img, width=1024) # coz costly to work on full resolution
				cropped, masked = roi(resized_image)
				rotated = straighten(cropped) # straighten the image
				zoomed = imutils.resize(rotated, width=1000)
				# masked = mask_color(zoomed)
				gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)

				# threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
				ret, threshold = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV) #barely works, but works
				img_rgb = cv2.cvtColor(threshold, cv2.COLOR_BGR2RGB)
				# img_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

				# cv2.imshow('window', resized_image)
				print('\r', f'frame: {frames}', end='')
				# if cv2.waitKey(25) & 0xFF == ord('q'):
				# 	break


				if CHECK:
					KILL_CHECK = check_kill(img_rgb, ign)
					if KILL_CHECK:
						KILL_FRAME = frames + 120
						click.secho(f'KILL FOUND at {frames}\nKILL FRAME is {KILL_FRAME}', fg='red')
						CHECK = False
					elif KILL_CHECK and frames < KILL_FRAME:
						CHECK = False
						continue

				if KILL_CHECK and frames == KILL_FRAME:
					make_clip()
					KILL_CHECK = False
					CHECK = True

			else:
				break

		del video
	except KeyboardInterrupt:
		pass
	except Exception as e:
		logger.error(e)
		raise e
	finally:
		click.secho('Finished processing.', fg='green')
		if CLIP_COUNTER == 0:
			click.secho('No clips found. :(', fg='red')
		else:
			click.secho(f'{CLIP_COUNTER} clips found.', fg='green')

		video.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
	del GLOBAL_BUFFER