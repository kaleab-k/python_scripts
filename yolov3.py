# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:40:49 2019

@author: Kaleab
"""

import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time

# def rotateRect(rect, heightPercentage, widthPercetange):
#     rwidth  = rect(3)
#     rheight = rect(4)
#     rect(3) = round((rwidth * widthPercetange) / 100)
#     rect(4) = round((rheight * heightPercentage) / 100)        
#     rect(1) += (rwidth - rect(3)) / 2
#     rect(2) += (rheight - rect(4)) / 2
#     return rect

width = 416
height = 416
inputs = tf.placeholder(tf.float32, [None, width, height, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
# model = nets.FasterRCNN(inputs)
#model = nets.YOLOv2(inputs, nets.Darknet19)

frameCount = 0

#frame=cv2.imread("D://pyworks//yolo//truck.jpg",1)

classes={'0':'person'}
list_of_classes=[0]
with tf.Session() as sess:
    sess.run(model.pretrained())
#"D://pyworks//yolo//videoplayback.mp4"   
    vid_dir = 'abandonedBox'
    path = '/home/kalex/Documents/MATLAB/PD/Videos/' + vid_dir + '/in%06d.jpg'
    cap = cv2.VideoCapture(path)
    list_file = open('%s.idl'%vid_dir, 'w')
    while(cap.isOpened()):
        ret, frame = cap.read()
        width_vid = cap.get(3)
        height_vid = cap.get(4)
        # print(width_vid)
        # print(height_vid)
        # width = 416
        # height = 416
        img=cv2.resize(frame,(int(width),int(height)))
        # img=cv2.resize(frame,(int(width),int(height)))
        imge=np.array(img).reshape(-1,int(height),int(height),3)
        start_time=time.time()
        preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
    	
        frameCount += 1
        list_file.write('"'+path % frameCount+'";')

        # print("--- %s seconds ---" % (time.time() - start_time)) 
        boxes = model.get_boxes(preds, imge.shape[1:3])
        # cv2.namedWindow('image',cv2.WINDOW_NORMAL)

        # cv2.resizeWindow('image', int(width),int(height))
        #print("--- %s seconds ---" % (time.time() - start_time)) 
        print("Frame #",frameCount)
        boxes1=np.array(boxes)
        for j in list_of_classes:
            count =0
            if str(j) in classes:
                lab=classes[str(j)]
            if len(boxes1) !=0:
                
                
                for i in range(len(boxes1[j])):
                    box=boxes1[j][i] 
                    #print(boxes1[j][i][4])
                    if boxes1[j][i][4]>=.3:    
                        count += 1 
                        Rx = width_vid/width
                        Ry = height_vid/height   
                        new_box = [0,0,0,0]
                        new_box[0] = int(Rx * box[0])
                        new_box[1] = int(Ry * box[1])
                        new_box[2] = int(Rx * box[2])
                        new_box[3] = int(Ry * box[3])
                        if(count == 1):
                            list_file.write(' (')
                        else:
                            list_file.write(', (')  
                        list_file.write(str(new_box[0])+', '+str(new_box[1])+', '+str(new_box[2]-new_box[0])+', '+str(new_box[3]-new_box[1])+ '):'+ str(boxes1[j][i][4]))
                        rect = (box[0], box[1], box[2]-box[0], box[3]-box[1]) 
                        # rotateRect(rect, 150, 150)
                        # res_img = cv2.resize(frame,(int(width_vid),int(height_vid)))
                        # cv2.rectangle(res_img,(new_box[0], new_box[1]),(new_box[2],new_box[3]),(0,255,0),1)
                        # cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
                        # cv2.imshow("res_image", res_img)
                        # cv2.rectangle(img,(rect(1),rect(2)),(rect(3)+rect(1),rect(4)+rect(2)),(255,0,0),1)
                        # cv2.putText(img, lab, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), lineType=cv2.LINE_AA)
            list_file.write(';\r\n')
            #print(lab,": ",count)
    
        # cv2.imshow("image",img)          
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break          

cap.release()
cv2.destroyAllWindows()    
