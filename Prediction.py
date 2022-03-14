import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

model = 'D:/User/Project/Code/Sign_Language_Translator/model.h5'
weights = 'D:/User/Project/Code/Sign_Language_Translator/pesos.h5'
cnn = load_model(model)
cnn.load_weights(weights)
address = 'D:/User/Project/Dataset/39words/Validacion'
dire_img = os.listdir(address)

# Read the camera
cap = cv2.VideoCapture(0)
hands_class = mp.solutions.hands
hands = hands_class.Hands()
draw = mp.solutions.drawing_utils

while 1:
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copy = frame.copy()
    result = hands.process(color)
    positions = []
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                height, width, c = frame.shape
                xPosition, yPosition = int(lm.x * width), int(lm.y * height)
                positions.append([id, xPosition, yPosition])
                draw.draw_landmarks(frame, hand, hands_class.HAND_CONNECTIONS)
            if len(positions) != 0:
                pto_i1 = positions[3]
                pto_i2 = positions[17]
                pto_i3 = positions[10]
                pto_i4 = positions[0]
                pto_i5 = positions[9]
                x1, y1 = (pto_i5[1] - 100), (pto_i5[2] - 100)
                width, height = (x1 + 200), (y1 + 200)
                x2, y2 = x1 + width, y1 + height
                fingers = copy[y1:y2, x1:x2]
                fingers = cv2.resize(fingers, (200, 200), interpolation=cv2.INTER_CUBIC)
                x = img_to_array(fingers)
                x = np.expand_dims(x, axis=0)
                vector = cnn.predict(x)
                result = vector[0]
                answer = np.argmax(result)
                print(result)
                if answer == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, 'ABRIL', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, 'ADULTO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 2:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, 'AGOSTO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 3:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, 'BIEN', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 4:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, 'BIENVENIDO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, 'CAMA', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 6:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
                    cv2.putText(frame, 'CINCO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 7:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'COMO_ESTAS', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 8:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'CON_MUCHO_GUSTO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 9:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'CUATRO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 10:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'DICIEMBRE', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 11:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'DOMINGO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 12:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'DOS', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 13:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'ENERO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 14:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'FEBRERO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 15:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'GRACIAS', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 16:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'HABITACION', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 17:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'HOLA', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 18:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'HOTEL', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 19:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'JUEVES', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 20:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'JULIO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 21:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'JUNIO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 22:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'LUNES', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 23:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'MAL', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 24:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'MARTES', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 25:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'MARZO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 26:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'MAYO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 27:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'MIERCOLES', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 28:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'NINO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 29:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'NO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 30:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'NOVIEMBRE', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 31:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'OCTUBRE', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 32:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'POR_FAVOR', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 33:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'SABADO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 34:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'SEPTIEMBRE', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 35:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'SI', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 36:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'TRES', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 37:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'UNO', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif answer == 38:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, 'VIERNES', (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)

                else:
                    cv2.putText(frame, 'SEÑA DESCONOCIDA', (x1, y1 - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
