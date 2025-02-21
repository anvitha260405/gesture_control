import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

# Capture Video
cap = cv2.VideoCapture(0)
last_action_time = time.time()
prev_distance = None  # Track previous pinch distance

def perform_action(action):
    global last_action_time
    if time.time() - last_action_time > 0.3:  # Adjust delay for better responsiveness
        if action == "left":
            pyautogui.press('left')
        elif action == "right":
            pyautogui.press('right')
        elif action == "zoom_in":
            pyautogui.hotkey('ctrl', '+')  # Single smooth zoom step
        elif action == "zoom_out":
            pyautogui.hotkey('ctrl', '-')  # Single smooth zoom step
        elif action == "pause":
            pyautogui.press('space')
        elif action == "resume":
            pyautogui.press('space')
        elif action == "thumbs_up":
            pyautogui.press('f5')
        elif action == "thumbs_down":
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
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract Landmark Points
            landmarks = hand_landmarks.landmark

            # Define Key Points
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            wrist = landmarks[0]

            # Swipe Gestures
            if index_tip.x < 0.2:
                perform_action("left")
            elif index_tip.x > 0.8:
                perform_action("right")
            elif thumb_tip.y < wrist.y - 0.15:
                perform_action("thumbs_up")
            elif thumb_tip.y > wrist.y + 0.15:
                perform_action("thumbs_down")

            # Pinch Zoom Gesture
            distance = ((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2) ** 0.5

            if prev_distance is not None:
                if distance < 0.03:  # If fingers are very close
                    perform_action("zoom_in")
                elif distance > 0.1:  # If fingers are far apart
                    perform_action("zoom_out")

            prev_distance = distance  # Store the previous distance for comparison

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
