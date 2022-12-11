import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
from pathlib import Path
import os
import ntpath
import pickle 

#dlib librairies pour faire de la reconnaissance faciale
if __name__ == '__main__':

    print("[INFO] programme FIT en cours d'exécution..")
    print("[INFO] Importation d'un modèle pré-entraîné..")
    #pose_predictor_5_point = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    pose_predictor_68_point = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    face_detector = dlib.get_frontal_face_detector()
    print("[INFO] Importation d'un modèle pré-entraîné..")

    #detection de visage
    #retourn une liste de coordonnées des visages sur l'image

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

    def transform(image, face_locations):
        coord_faces = []
        for face in face_locations:
            rect = face.top(), face.right(), face.bottom(), face.left()
            coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
            coord_faces.append(coord_face)
        return coord_faces


    print("[INFO] Importation de visages...")
    face_to_encode_path = Path('known_faces' )
    print(face_to_encode_path.exists == True)
    files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]

    for file_ in face_to_encode_path.rglob('*.png'):
        files.append(file_)
    if len(files)==0:
        raise ValueError('Aucun visage détecté dans le répertoire: {}'.format(face_to_encode_path))
    known_face_names = [os.path.splitext(ntpath.basename(file_))[0] for file_ in files]

    known_face_encodings = []
    for file_ in files:
        image = PIL.Image.open(file_)
        image = np.array(image)
        face_encoded = encode_face(image)[0][0]
        known_face_encodings.append(face_encoded)


    variable_known_face_names = open('known_face_names', 'wb') 
        
    pickle.dump(known_face_names, variable_known_face_names)  

    variable_known_face_names.close()


    variable_known_face_encodings = open('known_face_encodings', 'wb') 
        
    pickle.dump(known_face_encodings, variable_known_face_encodings)  

    variable_known_face_encodings.close()




