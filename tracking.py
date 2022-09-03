import cv2

img_file = 'car_image.jfif'
#video = cv2.VideoCapture('Tesla Dashcam Accident.mp4')
video = cv2.VideoCapture('Dashcam Pedestrians.mp4')

# pre trained car and pedestrian classifiers
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'


# creating car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


# run until car stops
while True:
    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        # converting to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # draw rectsngles around cars
    # car1 = cars[0]
    # (x, y, w, h) = car1
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw rectangles around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Display the image with the faces spotted
    cv2.imshow("Car and pedestrian detector", frame)

    # waiting
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Release the video capture
video.release()

