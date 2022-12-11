
from datetime import datetime
import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import argparse
from pathlib import Path
import os
import ntpath
import string
import random
import datetime
from PIL import ImageTk, Image
import pickle 

parser = argparse.ArgumentParser(description='Easy Facial Recognition App')
parser.add_argument('-i', '--input', type=str, required=True, help='directory of input known faces')

print('Démarrage du système')
print("[INFO] Importation d'un modèle pré-entraîné..")
#pose_predictor_5_point = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
pose_predictor_68_point = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
print("[INFO] Importation d'un modèle pré-entraîné..")


def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces


def encode_face(image):
    face_locations = face_detector(image, 1)
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # DETECT FACES
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
        # GET LANDMARKS
        shape = face_utils.shape_to_np(shape)
        landmarks_list.append(shape)
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list


def easy_face_reco(frame, known_face_encodings, known_face_names , image = False):
    if (image):rgb_small_frame = frame
    else: rgb_small_frame = frame[:, :, ::-1]
    # ENCODING FACE
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # CHECK DISTANCE BETWEEN KNOWN FACES AND FACES DETECTED
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "inconnu"
            
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)
    date = datetime.datetime.now()
    if (image): 
        cv2.imwrite('output/IMG-'+date.strftime("%d_%m_%Y_%H_%M_%S")+'.jpg',frame)
        cv2.imshow('output',frame)
        cv2.waitKey(0)
    else: cv2.imwrite('captures/IMG-'+date.strftime("%d_%m_%Y_%H_%M_%S")+'.jpg',frame)
   


if __name__ == '__main__':
    # args = parser.parse_args()
    choice = input('*****menu***** \n 1 => camera\n 2 => local image \n 0 => exit \n')
    if choice ==0: exit()
    # load the model from disk
    print('[INFO] bien importés')
    known_face_encodings = pickle.load(open('known_face_encodings', 'rb'))
    known_face_names = pickle.load(open('known_face_names', 'rb'))
    print('[INFO] bien importés')

    if choice == '2':
        img = cv2.imread("input/WIN_20221027_10_31_31_Pro.jpg")
        easy_face_reco(img, known_face_encodings, known_face_names , True)
    elif choice == '1':
        print('[INFO] Démarrage de la webcam...')
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("[Erreur] Impossible d'ouvrir l'appareil photo")
            exit()
        print('[INFO] Webcam bien démarrée')
        print('[INFO] Détection...')
        a =1
        while True:
            a+=1
            print(a)
            ret, frame = video_capture.read()
                # if frame is read correctly ret is True
            if not ret:
                print("Impossible de recevoir la trame (fin du flux ?). Sortie ...")
                break
            easy_face_reco(frame, known_face_encodings, known_face_names)
            cv2.imshow('Facial Recognition App', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('waitKey')
                print(cv2.waitKey(1) & 0xFF == ord('q'))
                break
        print('[INFO] Arrêt du système')
        video_capture.release()
        cv2.destroyAllWindows()
   
