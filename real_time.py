import cv2
import numpy as np
import pickle
import os


def real_time(model_path):
    model = pickle.load(open(model_path, 'rb'))
    classes = os.listdir('garbage_classification')
    cap = cv2.VideoCapture(1)

    cap.set(3, 1280)
    cap.set(4, 640)

    while True:
        ret, frame = cap.read()
        img = cv2.flip(frame, 1)

        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.putText(img, 'Put your face under here', (400, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
        cv2.rectangle(img, (400, 200), (800, 600), (255, 0, 0))

        face = img_[200:600, 400:800]
        face = face / 255.0
        face = cv2.resize(face, (128, 128))
        face = face.reshape((1, -1))
        pred = model.predict(face)
        name = np.squeeze(pred)

        cv2.putText(img, '{}'.format(classes[name]), (400, 640), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

real_time('models/best_svm_model.pkl')