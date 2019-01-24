#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
import sys
import cv2
class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    arg = sys.argv[1]
    img = cv2.imread(arg)
    img = cv2.resize(img, (1280, 720))

    boxes, scores, classes, num = odapi.processFrame(img)

    for i in range(len(boxes)):
        if classes[i]==1 and scores[i] > threshold:
            box = boxes[i]
            
            print("Body Detected")
            model_path_1 = 'frozen_inference_graph_face.pb'
            odapi1 = DetectorAPI(path_to_ckpt=model_path_1)
            threshold = 0.7
            box = boxes[i]
            cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),4)
            boxes1, scores1, classes1, num1 = odapi1.processFrame(img)
            for j in range(len(boxes1)):
                threshold1 = 0.57
                print(classes[j],"m,mda,ma")
                if(classes1[j]==1):
                    if scores1[j] > threshold1:
                        box1 = boxes1[j]
                        cv2.rectangle(img,(box1[1],box1[0]),(box1[3],box1[2]),(155,155,0),2)
                        cv2.putText(img,"Face Detected",(box1[1]+box1[0],box1[0]),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 125, 255), lineType=cv2.LINE_AA)

                else:
                    print(classes1[j])
                    cv2.putText(img,"Face Not Detected",(box[1]+box[0],box[0]),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 125, 255), lineType=cv2.LINE_AA)
                    
    
            
    cv2.imshow("preview", img)

    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




