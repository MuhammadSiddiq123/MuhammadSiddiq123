# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 08:28:29 2022

@author: SiddiQ
"""

import face_recognition
known_image = face_recognition.load_image_file(r"D:\images_set\akrammm.jpeg")
unknown_image = face_recognition.load_image_file(r"D:\images_set\akram.jpeg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(results)
