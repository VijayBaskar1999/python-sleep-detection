from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from playsound import playsound
import winsound





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
thresh = 0.25
frame_check = 40
total_yawn_count = 0
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")  # Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["jaw"]
cap = cv2.VideoCapture(0)
flag = 0
mflag=0
jflag=0
temp_yawn_count=0
yawned = False
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=450)
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
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        jawHull = cv2.convexHull(jaw)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "Eye_aspect_ratio:{:.2f}".format(ear), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Mouth_aspect_ratio:{:.2f}".format(mouthEAR), (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, "Head Slope:{:.2f}".format(head_slope), (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, "height_ratio:{:.2f}".format(jaw_h_ratio), (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "frame_rate:{:.2f}".format(frame_check), (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Total_yawn:{:.2f}".format(total_yawn_count), (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
               # playsound('alert.mp3')
                winsound.Beep(2500, 1000)
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            flag = 0
        if mouthEAR > 0.50:
            mflag +=1
            #print(flag)
            if mflag >= 17:
                yawned  =True

        else:
            if yawned:
                total_yawn_count += 1
                temp_yawn_count += 1
                if (temp_yawn_count == 2):
                    frame_check -= 2
                    temp_yawn_count -= 1
                yawned = False
            mflag = 0

        if head_slope < 3 or jaw_h_ratio < 0.55 or jaw_h_ratio > 0.84:
            jflag += 1
            #print(jflag)
            if jflag >= frame_check:
                # playsound('alert.mp3')
                winsound.Beep(2500, 1000)
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            jflag = 0
        if total_yawn_count>=5:
            cv2.putText(frame, "************YOU SEEMS SLEEPY**********", (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()