import time
import cv2

# Load the pre-trained Haar Cascade classifier for face detection
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Access the default camera (usually the webcam)
cam = cv2.VideoCapture(0)
time.sleep(1)  # Allow the camera to warm up

face_detected = False  # Flag to track face detection
previous_state = None  # Flag to track the previous detection state

while True:
    _, img = cam.read()  # Read a frame from the camera
    
    # Convert the frame to grayscale for face detection
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    face = haar_cascade.detectMultiScale(grayimg)
    
    # Check if a face is detected
    if len(face) > 0:
        if previous_state != "detected":
            print("Face detected")
            previous_state = "detected"
        text = "Face detected"
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        if previous_state != "not detected":
            print("NO face detected")
            previous_state = "not detected"
        text = "NO face detected"
    
    # Display the text message on the video feed
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the frame with the detected faces
    cv2.imshow("Face Detection", img)
    
    # Wait for 10 milliseconds for a key press
    key = cv2.waitKey(10)
    
    # Break the loop if the 'q' key is pressed
    if key == ord("q"):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
