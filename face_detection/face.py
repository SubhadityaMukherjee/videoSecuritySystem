import cv2
import time

def crop(videopath):

	vid = cv2.VideoCapture(0)

	count = 0

	success = 1

	while success:

		success, image = vid.read()
		cv2.imshow('image',image)
		k=cv2.waitKey(20)
		if k==27:
			break

	# image = cv2.imread(imagepath)
	# image = cv2.resize(image,(600,600))
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	faceCascade = cv2.CascadeClassifier( "haarcascade_frontalface_default.xml")
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.3,
		minNeighbors=3,
		minSize=(30,30)
		)
	ts=time.time()
	count=0

	for(x,y,w,h) in faces:
		crop = image[y:y+h, x:x+w]
		timestamp="{}({}).jpg".format(ts,count)
		cv2.imwrite(timestamp, crop)
		count+=1


crop("IMG_20181215_143031.jpg")