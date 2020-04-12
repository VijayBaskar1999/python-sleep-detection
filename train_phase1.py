import imutils
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import dlib
from imutils import face_utils
import csv

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh=0.25
flag=0


DATADIR = "C:/Users/vijay/PycharmProjects/ML/morse/training"
CATEGORIES = ["sleeping", "notsleeping"]
vid_array=[]
class_num=[]
thresh_array=[]
frame_array=[]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    print(path)
    class_num.append(CATEGORIES.index(category))
    for vid in os.listdir(path):
        vid_array.append(cv2.VideoCapture(os.path.join(path, vid)))

for videos in vid_array:
    min_thresh = thresh
    max_frame = 0
    while(videos.isOpened()):
        try:
            ret, frame = videos.read()
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=640, height=480)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < thresh:
                    min_thresh=ear
                    flag += 1
                    #print(flag)
                    if flag > max_frame:
                        max_frame=flag
                else:
                    flag = 0
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        except AttributeError:
               break

    thresh_array.append(min_thresh)
    frame_array.append(max_frame)
    del frame
    videos.release()
    cv2.destroyAllWindows()

#print(vid_array)
#print(class_num)
#print(thresh_array)
#print(frame_array)
training_data=[]

with open('dataset_phase1.csv','w',newline='') as csvfile:
    fieldname=['Ground Truth','video','filename','minimum_thresh','frames(no of frames eye closed)']
    thewriter=csv.DictWriter(csvfile,fieldnames=fieldname)
    thewriter.writeheader()
    i=0

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        print(path)
        class_num.append(CATEGORIES.index(category))
        for vid in os.listdir(path):
            thewriter.writerow({'Ground Truth':CATEGORIES.index(category), 'video':vid_array[i],'filename':vid, 'minimum_thresh':thresh_array[i], 'frames(no of frames eye closed)':frame_array[i]})
            i+=1
print("Dataset Created Successfully!!!")
