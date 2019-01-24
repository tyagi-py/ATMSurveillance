#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
import numpy as np
from keras.applications import vgg16,inception_v3,resnet50,mobilenet


# In[3]:


vgg_model = vgg16.VGG16(weights='imagenet')


# In[4]:


from keras.preprocessing.image import load_img
# image = load_img('plier.jpeg', target_size=(224, 224))


# In[5]:


from keras.preprocessing.image import img_to_array
# image = img_to_array(image)


# In[6]:


# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))


# In[7]:


from keras.applications.vgg16 import preprocess_input
# image = preprocess_input(image)


# In[8]:


# yhat = vgg_model.predict(image)


# In[9]:


from keras.applications.vgg16 import decode_predictions
# label = decode_predictions(yhat)
# 9= label[0][0]
# print('%s (%.2f%%)' % (l[1], l[2]*100))


# In[21]:


import sys
import cv2
if __name__ == '__main__':
    arg = sys.argv[1]
    try:
        arg = int(arg)
    except:
        pass
    cap = cv2.VideoCapture(arg)
    while True:
        f,img = cap.read()
        image = cv2.resize(img,(224,224),cv2.INTER_AREA)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = vgg_model.predict(image)
        label = decode_predictions(yhat)
        l = label[0][0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "DANGER"

    # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        textX = int((img.shape[1] - textsize[0]) / 2)
        textY = int((img.shape[0] + textsize[1]) / 2)
        if l[2]>0.2:
            print('%s (%.2f%%)' % (l[1], l[2]*100))
            if(l[1]=='screwdriver' or l[1]=='hammer' or l[1]=='power_drill' or l[1]=='chain_saw'):
                cv2.putText(img,text+' '+l[1],(textX,textY),font,1,(0,0,255),2)
        cv2.imshow("frame",img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:




