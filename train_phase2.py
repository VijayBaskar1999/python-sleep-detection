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
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mou = (A + B) / (2.0 * C)
    return mou

def calc_head_slope(jaw):
    midpoint = [(jaw[0][0] + jaw[16][0]) / 2, (jaw[0][1] + jaw[16][1]) / 2]
    slope = (jaw[8][1]-midpoint[1])/(jaw[8][0]-midpoint[0])
    return abs(slope)
def calc_jaw_h_ratio(jaw):
    midpoint = [(jaw[0][0] + jaw[16][0]) / 2, (jaw[0][1] + jaw[16][1]) / 2]
    h = distance.euclidean(midpoint, jaw[8])
    b = distance.euclidean(jaw[0],jaw[16])
    return h/b

thresh=0.25
flag=0
mflag=0
jflag=0
temp_yawn_count=0
yawned = False



DATADIR = "C:/Users/vijay/PycharmProjects/ML/morse/training"
CATEGORIES = ["sleeping", "notsleeping"]
vid_array=[]
class_num=[]
thresh_array=[]
eye_frame_array=[]
yawn_count=[]
head_frame_array=[]


for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    print(path)
    class_num.append(CATEGORIES.index(category))
    for vid in os.listdir(path):
        vid_array.append(cv2.VideoCapture(os.path.join(path, vid)))

for videos in vid_array:
    min_thresh = thresh
    max_frame = 0
    head_max_frame = 0
    total_yawn_count = 0
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
                mouth = shape[mStart:mEnd]
                jaw = shape[jStart:jEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                mouthEAR = mouth_aspect_ratio(mouth)
                ear = (leftEAR + rightEAR) / 2.0
                head_slope = calc_head_slope(jaw)
                jaw_h_ratio = calc_jaw_h_ratio(jaw)

                if ear < thresh:
                    min_thresh=ear
                    flag += 1
                    #print(flag)
                    if flag > max_frame:
                        max_frame=flag
                else:
                    flag = 0
                if mouthEAR > 0.50:
                    mflag += 1
                    # print(flag)
                    if mflag >= 17:
                        yawned = True

                else:
                    if yawned:
                        total_yawn_count += 1
                        yawned = False
                    mflag = 0

                if head_slope < 3 or jaw_h_ratio < 0.55 or jaw_h_ratio > 0.84:
                    jflag += 1
                    #print(flag)
                    if jflag > head_max_frame:
                        head_max_frame=jflag
                else:
                    jflag = 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        except AttributeError:
               break

    thresh_array.append(min_thresh)
    eye_frame_array.append(max_frame)
    yawn_count.append(total_yawn_count)
    head_frame_array.append(head_max_frame)
    del frame
    videos.release()
    cv2.destroyAllWindows()

#print(vid_array)
#print(class_num)
#print(thresh_array)
#print(frame_array)
training_data=[]

with open('dataset_phase2.csv','w',newline='') as csvfile:
    fieldname=['Ground Truth','video','filename','minimum_thresh','frames(no of frames eye closed)','yawn count','headframe(no of frames head down)']
    thewriter=csv.DictWriter(csvfile,fieldnames=fieldname)
    thewriter.writeheader()
    i=0

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        print(path)
        class_num.append(CATEGORIES.index(category))
        for vid in os.listdir(path):
            thewriter.writerow({'Ground Truth':CATEGORIES.index(category), 'video':vid_array[i],'filename':vid, 'minimum_thresh':thresh_array[i], 'frames(no of frames eye closed)':eye_frame_array[i], 'yawn count':yawn_count[i],'headframe(no of frames head down)':head_frame_array[i]})
            i+=1
print("Dataset Created Successfully!!!")
