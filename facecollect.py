import cv2
import numpy as np
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
skip=0
facedata=[]
datapath='C:\\Users\\Bonesnatcher\\Desktop\\face_recognition\\data\\'
filename=input("enter name")
while True:
	ret,frame=cap.read()
	if ret==False:
		continue
	faces=face_cascade.detectMultiScale(frame,1.5,3)
	faces=sorted(faces,key=lambda f:f[2]*f[3])
	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),2)
		skip+=1
		offset=10
		facesection=frame[y-offset:y+w+offset,x-offset:x+h+offset]
		facesection=cv2.resize(facesection,(100,100))
		if skip%10==0:
			facedata.append(facesection)
			print(len(facedata))

	cv2.imshow('video',frame)
	cv2.imshow('Video',facesection)
	key_pressed=cv2.waitKey(1) &0XFF
	if key_pressed==ord('q'):
		break	
facedata=np.asarray(facedata)
facedata=facedata.reshape((facedata.shape[0],-1))
print(facedata.shape)

np.save(datapath+filename+'.npy',facedata)
print("data successfully saved at"+datapath+filename+'.npy')
cap.release()
cv2.destroyAllWindows()	