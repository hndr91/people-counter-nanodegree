import numpy as np
import tensorflow as tf

class TFInferece:

    def __init__(self):
        self.model_path = None
        self.detection_graph = None
        self.default_graph = None
        self.sess = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None
    
    def load_model(self, model_path):
        
        self.model_path = model_path

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')  

        return
    
    def exec_session(self):
        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # input
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        # output
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        return

    def exec_detection(self, frame):
        image_np_expanded = np.expand_dims(frame, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded}
        )

        height, width,_ = frame.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * height), int(boxes[0,i,1] * width), int(boxes[0,i,2] * height), int(boxes[0,i,3] * width))

        
        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

