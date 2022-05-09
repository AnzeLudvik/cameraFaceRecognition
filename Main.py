import cv2
from random import randrange

#load pre-trained data from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#video capture from webcam (0 = is the main webcam on your pc/laptop)
webcam = cv2.VideoCapture(0)

#looping over frames
while True:

    #read current frame
    successful_frame_read, frame = webcam.read()

    #convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Face detection
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #face square
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
    
    #display
    cv2.imshow("Webcam face recognition", frame)
    key = cv2.waitKey(1)

    #Press ESC to stop
    if key==27:
        break

webcam.release()
print('Code completed')