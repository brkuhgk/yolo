#!/usr/bin/env python
# coding: utf-8

# In[193]:


import cv2
import numpy as np
import os
import math
import time


# In[201]:


#load yolo3
start =time.time()

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

end= time.time()
print("To load yolo3 time :",end-start)


# In[195]:


#vediocapture
vediocap = cv2.VideoCapture('chair2.mp4')
vediocap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
vediocap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
vediocap.set(cv2.CAP_PROP_FPS, 30) 

writer = None
(height,width) = (None, None)





start =time.time()




while vediocap.isOpened():
    ret, frame = vediocap.read()
    
    if not ret:
        break
    
    #if frame dimensions are empty
    if width is None or height is None: 
        (height, width) = frame.shape[:2]
    
    #320×320 it’s small so less accuracy but better speed
    #609×609 it’s bigger so high accuracy and slow speed
    #416×416 it’s in the middle and you get a bit of both.
    
    start =time.time()
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    outs = net.forward(output_layers)
    
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    #to remove the noise
    #Non maximum suppresion.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #0.5 =confidence ,Threshold =0.4
    
    
    
    #atleast one is detection exists
    if(len(indexes) >0):
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
        
    
    
    
    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output_1.avi", fourcc, 30,(frame.shape[1], frame.shape[0]), True)
        
    writer.write(frame)
#   cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
        

vediocap.release()
outer.release()
cv2.destroyAllWindows()
end =time.time()
print("time taken : ",end -start)






