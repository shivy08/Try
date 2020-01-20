import numpy as np
import cv2
import tensorflow.keras

import pygame, random, sys
from pygame.locals import *
def collide(x1, x2, y1, y2, w1, w2, h1, h2):
	if x1+w1>x2 and x1<x2+w2 and y1+h1>y2 and y1<y2+h2:return True
	else:return False
def die(screen, score):
	f=pygame.font.SysFont('Arial', 30);t=f.render('Your score was: '+str(score), True, (0, 0, 0));screen.blit(t, (10, 270));pygame.display.update();pygame.time.wait(2000);sys.exit(0)
xs = [290, 290, 290, 290, 290];ys = [290, 270, 250, 230, 210];dirs = 0;score = 0;applepos = (random.randint(0, 590), random.randint(0, 590));pygame.init();s=pygame.display.set_mode((600, 600));pygame.display.set_caption('Snake');appleimage = pygame.Surface((10, 10));appleimage.fill((0, 255, 0));img = pygame.Surface((20, 20));img.fill((255, 0, 0));f = pygame.font.SysFont('Arial', 20);clock = pygame.time.Clock()


np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('directions.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cap = cv2.VideoCapture(0)
while True:

	ret, frame = cap.read()
	resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
	resized = np.fliplr(resized)
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame', gray)
	normalized_image_array = (resized.astype(np.float32) / 127.0) - 1
	data[0] = normalized_image_array
	prediction = model.predict(data)
	#dict = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
	prediction = prediction[0]
	value = max(prediction)
	label = list(prediction).index(value)
	#x = dict[label]
	if value >= 0.97: print(label, value)


	clock.tick(2)
	for e in pygame.event.get():
		if e.type == QUIT:
			sys.exit(0)
		elif value>=0.97:
			if label == 0 and dirs != 0:dirs = 2
			elif label==1 and dirs != 2:dirs = 0
			elif label==2 and dirs != 1:dirs = 3
			elif label==3 and dirs != 3:dirs = 1
	i = len(xs)-1
	while i >= 2:
		if collide(xs[0], xs[i], ys[0], ys[i], 20, 20, 20, 20):die(s, score)
		i-= 1
	if collide(xs[0], applepos[0], ys[0], applepos[1], 20, 10, 20, 10):score+=1;xs.append(700);ys.append(700);applepos=(random.randint(0,590),random.randint(0,590))
	if xs[0] < 0 or xs[0] > 580 or ys[0] < 0 or ys[0] > 580: die(s, score)
	i = len(xs)-1
	while i >= 1:
		xs[i] = xs[i-1];ys[i] = ys[i-1];i -= 1
	if dirs==0:ys[0] += 20
	elif dirs==1:xs[0] += 20
	elif dirs==2:ys[0] -= 20
	elif dirs==3:xs[0] -= 20	
	s.fill((255, 255, 255))	
	for i in range(0, len(xs)):
		s.blit(img, (xs[i], ys[i]))
	s.blit(appleimage, applepos);t=f.render(str(score), True, (0, 0, 0));s.blit(t, (10, 10));pygame.display.update()
cap.release()
cv2.destroyAllWindows()
					
			


