import cv2
import os
import numpy as np
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
def distance(x1,x2):
	return np.sqrt(((x1-x2)**2).sum())
def knn(training_set,test,k=5):
	ix=trainset[:,:-1]
	iy=trainset[: ,-1]
	m=trainset.shape[0]
	vals=[]
	for i in range(m):
		d=distance(ix[i],test)
		vals.append([d,iy[i]])
	vals=sorted(vals)
	vals=vals[:k]

	new_vals=np.unique(vals,return_counts=True)
	#print(new_vals.shape)

	index=new_vals[1].argmax()
	predict=new_vals[0][index]
	return predict

name={ }
class_id=0
face_data=[]


labels=[]
data_path='C:\\Users\\Bonesnatcher\\Desktop\\face_recognition\\data\\'
for fx in os.listdir(data_path):
	if fx.endswith('.npy'):
		name[class_id]=fx[:-4]
		data=np.load(data_path+fx)
		face_data.append(data)
		
		#create labels 
		target=class_id*np.ones((data.shape[0],1))
		labels.append(target)
		class_id+=1
face_dataset=np.concatenate(face_data,axis=0)
label_dataset=np.concatenate(labels,axis=0)

#print(face_dataset.shape)
#print(label_dataset.shape)

trainset=np.concatenate((face_dataset,label_dataset),axis=1)
print(trainset.shape)

while True:
	ret,frame=cap.read()

	if ret==False:
		continue
	faces=face_cascade.detectMultiScale(frame,1.5,3)
	for face in faces:
		x,y,w,h=face

		
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))

		out=knn(trainset,face_section.flatten())

		#predicted name
		pred_name=name[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(250,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

	cv2.imshow('video',frame)
	key_pressed=cv2.waitKey(1) &0XFF
	if key_pressed==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()