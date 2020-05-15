import numpy as np
import tensorflow as tf
import cv2
from imutils.video import FPS
import time

class PeopleDetectorAPI:

    def __init__(self, model_path):
        self.model_path = model_path
        
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
    def drawFrame(self, frame):
        image_np_expanded = np.expand_dims(frame, axis=0)

        # start_time = time.perf_counter()

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded}
        )

        # stop_time = time.perf_counter()
        # elapsed_time = stop_time - start_time
        
        # print("Elapsed Time : {}".format(self.fps.fps()))

        height, width,_ = frame.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * height), int(boxes[0,i,1] * width), int(boxes[0,i,2] * height), int(boxes[0,i,3] * width))

        
        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close_sess(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = "/home/rm6-polinema/Documents/tf-models/ssd_mobilenet_v2_coco/frozen_inference_graph.pb"
    video_path = "/home/rm6-polinema/Documents/nanodegree/pj01/nd131-openvino-fundamentals-project-starter/resources/Pedestrian_Detect_2_1_1.mp4"
    detector = PeopleDetectorAPI(model_path)
    th = 0.5

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.5
    color = (255,0,0)
    thk = 1

    cap = cv2.VideoCapture(video_path)
    cap.open(video_path)

    while cap.isOpened():
        fps = FPS().start()
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(30)

        boxes, scores, classes, num = detector.drawFrame(frame)

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > th:
                box = boxes[i]
                cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255,0,0), 2)
        
        fps.update()
        fps.stop()
        infer = "Approx FPS : {:.2f}".format(fps.fps())
        cv2.putText(frame, infer, (5, 20), font, fontscale, color, thk, cv2.LINE_AA, False)

        cv2.imshow('capture', frame)

        if key_pressed == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    