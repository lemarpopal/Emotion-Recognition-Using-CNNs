import os
import cv2
import numpy as np
import keras
import tensorflow as tf
import mtcnn
from gaze_tracking import GazeTracking

def main():
    ### SET OPTIONS
    """
    show_frame: True or False. Set to true if you want to see the video/webcam.
    video_path: String. Path to video file to play.
    source_path: String. Path to video to run emotion detector on, or just 'webcam' to use webcam.
    save_output: Save output of detected emotions video to a file called "output".
    save_log: String. Save log file of emotions to a text file. Specify log file name or leave empty string for none.
    """

    show_frame = True
    video_path = 'test_video.mov'
    source_path = 'webcam'
    save_output = True
    save_log = ""

    ###

    count = 0 # frame counter

    # load the model
    model = tf.keras.models.load_model(
        'tl-weights-improvement-08-0.69.hdf5',
        custom_objects=None,
        compile=False
    )

    if source_path != "webcam":
        webcam=cv2.VideoCapture(source_path)
    else:
        webcam=cv2.VideoCapture(0)

    video=cv2.VideoCapture(video_path)

    if save_output:
        # Used for writing video file
        frame_width = int(webcam.get(3))
        frame_height = int(webcam.get(4))
        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1000, 600))

    # initialize face detector 
    detector = mtcnn.MTCNN()

    # initialize gazer tracker
    gaze = GazeTracking()
    while True:
        ret,frame = webcam.read() # captures frame and returns boolean value and captured image

        vret, vframe = video.read()

        # if either of the videos end, 
        if not ret or not vret:
            break

        print("Frame:", count)
        count += 1

        gray_img = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gaze.refresh(frame)
        frame = gaze.annotated_frame()
        where_looking = is_distracted(gaze, gray_img)

        if where_looking == "Distracted":
            cv2.putText(frame, "Distracted", (int(90), int(60)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            resized_img = cv2.resize(frame, (1000, 600))

            if show_frame:
                resized_vframe = cv2.resize(vframe, (500,300))
                cv2.imshow(video_path, resized_vframe)

                cv2.imshow('Facial emotion analysis', resized_img)
                
            if save_output:
                out.write(resized_img)

            if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
                break
            
            continue
        
        # write eye location details
        cv2.putText(frame, where_looking, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31),1)

        # use MTCNN to detect location of faces in image
        results = detector.detect_faces(gray_img)

        if len(results) != 0:
            x1, y1, width, height = results[0]['box']
            faces_detected = [(x1, y1, width, height)]


            for (x,y,w,h) in faces_detected:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
                roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
                roi_gray=cv2.resize(roi_gray,(64,64))
                img_pixels = keras.preprocessing.image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                #find max indexed array
                max_index = np.argmax(predictions[0])

                emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
                predicted_emotion = emotions[max_index]

                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        resized_img = cv2.resize(frame, (1000, 600))

        if show_frame:
            resized_vframe = cv2.resize(vframe, (500,300))
            cv2.imshow(video_path, resized_vframe)

            cv2.imshow('Facial emotion analysis',resized_img)
        
        if save_output:
            out.write(resized_img)

        if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
            break

    webcam.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows

def is_distracted(gaze, frame):

    # We send this frame to GazeTracking to analyze it
    # gaze.refresh(frame)

    # frame = gaze.annotated_frame()

    if gaze.is_blinking():
        return "Eyes Closed"
    elif gaze.is_right():
        return "Looking right"
    elif gaze.is_left():
        return "Looking left"
    elif gaze.is_center():
        return "Looking center"
    elif gaze.is_distracted():
        return "Distracted"

    return None

if __name__ == '__main__':
    main()