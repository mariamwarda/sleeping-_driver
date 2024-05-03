from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import time


ALARM_ON = False
model_path = "shape_predictor_68_face_landmarks.dat"


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    area = (A + B) / (2.0 * C)
    return area


def detect():
    er = 0.25
    frame_check = 10  # default is 20
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor(model_path)  

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)
    flag = 0
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640, height=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)  
            leftEye = shape[lStart:lEnd]
            print("leftEye",leftEye)
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            area = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            if area < er:
                flag += 1
                if flag >= frame_check:
                    if not ALARM_ON:
                        ALARM_ON = True
                        time.sleep(0.01)
                        
                    cv2.putText(frame, "****************Drowsy!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame,'sleeping time 00:'+str(flag)+'sec', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
            else:
                flag = 0
                ALARM_ON = False
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    detect()

