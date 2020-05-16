import numpy as np
import cv2
from argparse import ArgumentParser
from tf_inference import TFInferece
from imutils.video import FPS

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.5
color = (255,0,0)
thk = 1
fps_list = []


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to a pb frozen model.")
    parser.add_argument("-i", "--input", required=False, default="WEBCAM",
                        type=str, help="WEBCAM or path to video file.")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="Minimum detection probability threshold")
    
    return parser

def infer(args):
    tf_infer = TFInferece()
    
    prob_threshold = args.threshold

    tf_infer.load_model(args.model)
    tf_infer.exec_session()

    if "WEBCAM" in args.input:
        cap = cv2.VideoCapture(0)
        cap.open(0)
    else:
        cap = cv2.VideoCapture(args.input)
        cap.open(args.input)
    

    while cap.isOpened():
        fps = FPS().start()
        flag, frame = cap.read()

        if not flag:
            break
        key_pressed = cv2.waitKey(30)

        boxes, scores, classes, num = tf_infer.exec_detection(frame)

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > prob_threshold:
                box = boxes[i]
                cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), color, thk)

        fps.update()
        fps.stop()
        fps_list.append(fps.fps())
        infer = "Approx FPS : {:.2f}".format(fps.fps())
        cv2.putText(frame, infer, (5, 20), font, fontscale, color, thk, cv2.LINE_AA, False)

        cv2.imshow('capture', frame)

        if key_pressed == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Average FPS : {}".format(np.mean(fps_list)))

def main():
    args = build_argparser().parse_args()
    infer(args)

if __name__ == '__main__':
    main()