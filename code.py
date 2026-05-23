import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0
color = (255,0,255)  # default purple


def fingers_up(hand):
    tips = [4,8,12,16,20]
    fingers = []

    # thumb
    if hand.landmark[4].x > hand.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # other fingers
    for tip in tips[1:]:
        if hand.landmark[tip].y < hand.landmark[tip-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(hand_landmarks)

            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            # 1 finger -> draw
            if fingers == [0,1,0,0,0]:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas,(prev_x,prev_y),(x,y),color,8)
                prev_x, prev_y = x, y

            # 2 fingers -> blue
            elif fingers == [0,1,1,0,0]:
                color = (255,0,0)
                prev_x, prev_y = 0,0

            # 3 fingers -> green
            elif fingers == [0,1,1,1,0]:
                color = (0,255,0)
                prev_x, prev_y = 0,0

            # fist -> erase
            elif fingers == [0,0,0,0,0]:
                cv2.circle(canvas,(x,y),40,(0,0,0),-1)

            # open palm -> clear canvas
            elif fingers == [1,1,1,1,1]:
                canvas = np.zeros_like(frame)
                prev_x, prev_y = 0,0

    else:
        prev_x, prev_y = 0,0

    frame = cv2.add(frame,canvas)

    cv2.imshow("Aadya Gesture Paint",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
