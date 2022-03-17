# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 08:28:29 2022
images verification/face matching through python and face_recognition model
@author: SiddiQ
"""

import face_recognition     # importing the face recognition model

# In the below two line we just passing the paths of the images that we are going to verify
Img1_Path = face_recognition.load_image_file(r"D:\images_set\akrammm.jpeg")
Img2_Path = face_recognition.load_image_file(r"D:\images_set\akram.jpeg")

# The below two lines of code will just encode the images from there paths
Img1_encoding = face_recognition.face_encodings(Img1_Path)[0]
Img2_encoding = face_recognition.face_encodings(Img1_Path)[0]

# the below line will just compare the two encoded images weather it matchs or not
results = face_recognition.compare_faces([Img1_encoding], Img2_encoding)
print(results)
