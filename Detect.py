import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from servo import *
import RPi.GPIO as GPIO
from Motor import *



def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  counter, fps = 0, 0
  start_time = time.time()
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 255, 0)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  PWM=Motor()  
  servo=Servo()

  lst = [1,3,1,3,1,3,1,3]
  for i in lst:
      if i==1:
        PWM.setMotorModel(1000,1000,1000,1000) 
        time.sleep(1)
        PWM.setMotorModel(0,0,0,0)
      if i==2:
        PWM.setMotorModel(-1000,-1000,-1000,-1000) #back
        time.sleep(1)
        PWM.setMotorModel(0,0,0,0)
      if i==3:
        PWM.setMotorModel(1000,1000,-500,-500) #left
        time.sleep(1)
        PWM.setMotorModel(0,0,0,0)
      if i==4:
        PWM.setMotorModel(-500,-500,1000,1000) #right
        time.sleep(1)
        PWM.setMotorModel(0,0,0,0)      

      for i in range(30,151,60): #
        
        servo.setServoPwm('0',i)
        time.sleep(1)

        success, image = cap.read()
        time.sleep(1) 
        
        if not success:
            sys.exit(
            'ERROR: Unable to read from webcam. Please verify your webcam settings.'
                )

        counter += 1
        image = cv2.flip(image, 1)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        detection_result = detector.detect(input_tensor)

        image = utils.visualize(image, detection_result)

        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
