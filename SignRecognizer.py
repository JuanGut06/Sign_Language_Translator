import cv2
import mediapipe as mp
import os

signName = ""
address = 'D:/User/Project/Dataset/39words/Validacion/' + signName

if not os.path.exists(address):
    print('Carpeta creada: ', address)
    os.makedirs(address)

cont = 500
countLimit = 1000

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
        for mano in result.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                height, width, c = frame.shape
                xPosition, yPosition = int(lm.x*width), int(lm.y*height)
                positions.append([id, xPosition, yPosition])
                draw.draw_landmarks(frame, mano, hands_class.HAND_CONNECTIONS)
            if len(positions) != 0:
                pto_i1 = positions[4]
                pto_i2 = positions[20]
                pto_i3 = positions[12]
                pto_i4 = positions[0]
                pto_i5 = positions[9]
                
                x1, y1 = (pto_i5[1]-100),(pto_i5[2]-100)
                width, height = (x1+200),(y1+200)
                x2, y2 = x1 + width, y1 + height
                fingers = copy[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
            fingers = cv2.resize(fingers, (200,200), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(carpeta + "/" + signName + "_{}.jpg". format(cont), fingers)
            print(str(cont))
            cont = cont + 1

    cv2.imshow("Video", frame)

    k = cv2.waitKey(1)
    if k == 27 or cont >= countLimit:
            break

cap.release()
cv2.destroyAllWindows()
