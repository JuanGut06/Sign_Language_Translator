import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

modelo = 'C:/Users/juan7/Desktop/Juan-IA/ModelO.h5'
peso = 'C:/Users/juan7/Desktop/Juan-IA/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(peso)

direccion = 'C:/Users/juan7/Desktop/Juan-IA/Img/Validacion'
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

# LEEMOS LA CAMARA
cap = cv2.VideoCapture(0)
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()

dibujo = mp.solutions.drawing_utils

while (1):
    ret,frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []


    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                pto_i1 = posiciones[3]
                pto_i2 = posiciones[17]
                pto_i3 = posiciones[10]
                pto_i4 = posiciones[0]
                pto_i5 = posiciones[9]
                x1, y1 = (pto_i5[1] - 100), (pto_i5[2] - 100)
                ancho, alto = (x1 + 200), (y1 + 200)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                x = img_to_array(dedos_reg)
                x = np.expand_dims(x, axis=0)
                vector = cnn.predict(x)
                resultado = vector[0]
                respuesta = np.argmax(resultado)
                print(resultado)
                if respuesta == 0:
                    print("ADULTO")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, 'ADULTO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 1:
                    print("BIEN")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, 'BIEN', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 2:
                    print("BIENVENIDO")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, 'BIENVENIDO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 3:
                    print("CAMA")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, 'CAMA', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 4:
                    print("CINCO")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
                    cv2.putText(frame, 'CINCO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 5:
                    print("COMO_ESTAS")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'COMO_ESTAS', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 6:
                    print("CON_GUSTO")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'CON_GUSTO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 7:
                    print("CUATRO")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'CUATRO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 8:
                    print("DOS")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'DOS', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 9:
                    print("GRACIAS")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'GRACIAS', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 10:
                    print("HABITACIÓN")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'HABITACION', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 11:
                    print("HOLA")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'HOLA', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 12:
                    print("HOTEL")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'HOTEL', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 13:
                    print("MAL")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'MAL', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 14:
                    print("NINO_DOS")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'NINO_DOS', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)

                elif respuesta == 15:
                    print("NINO_UNO")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'NINO_UNO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)

                elif respuesta == 16:
                    print("NO")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'NO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 17:
                    print("POR_FAVOR")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'POR_FAVOR', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 18:
                    print("SI")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'SI', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 19:
                    print("TRES")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'TRES', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 20:
                    print("UNO")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'UNO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)

                else:
                    print("UNDEFINED")
                    cv2.putText(frame, 'LETRA DESCONOCIDA', (x1, y1 - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
