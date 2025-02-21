import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Capture Video
cap = cv2.VideoCapture(0)
last_action_time = time.time()
distance_history = []
hand_positions = {}
prev_distance = None

def perform_action(action):
    global last_action_time
    if time.time() - last_action_time > 0.3:  # Reduced delay for smoother actions
        if action == "left":
            pyautogui.press('left')
        elif action == "right":
            pyautogui.press('right')
        elif action == "zoom_in":
            pyautogui.hotkey('ctrl', '+')
        elif action == "zoom_out":
            pyautogui.hotkey('ctrl', '-')
        elif action == "pause":
            pyautogui.press('space')
        elif action == "resume":
            pyautogui.press('space')
        elif action == "start_slideshow":
            pyautogui.press('f5')
        elif action == "exit":
            pyautogui.press('esc')
        last_action_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_list = []
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            wrist = landmarks[0]
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            pinky_tip = landmarks[20]

            hand_list.append((wrist, index_tip, thumb_tip, pinky_tip))

            if hand_idx not in hand_positions:
                hand_positions[hand_idx] = wrist.x
            movement = wrist.x - hand_positions[hand_idx]

            if wrist.x > 0.8:
                perform_action("right")
                hand_positions[hand_idx] = wrist.x
            elif wrist.x < 0.2:
                perform_action("left")
                hand_positions[hand_idx] = wrist.x

            if thumb_tip.y < wrist.y - 0.15:
                perform_action("start_slideshow")
            elif thumb_tip.y > wrist.y + 0.15:
                perform_action("exit")

        if len(hand_list) == 1:
            wrist, index_tip, _, _ = hand_list[0]
            laser_x = int(index_tip.x * pyautogui.size()[0])
            laser_y = int(index_tip.y * pyautogui.size()[1])
            pyautogui.moveTo(laser_x, laser_y, duration=0.05)

        elif len(hand_list) == 2:
            _, index1, _, _ = hand_list[0]
            _, index2, _, _ = hand_list[1]

            distance = ((index1.x - index2.x) ** 2 + (index1.y - index2.y) ** 2) ** 0.5

            if prev_distance is not None:
                if distance > prev_distance + 0.02:  # Fingers moving apart
                    perform_action("zoom_in")
                elif distance < prev_distance - 0.02:  # Fingers moving closer
                    perform_action("zoom_out")

            prev_distance = distance  # Update previous distance

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
