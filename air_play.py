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
prev_distance = None
laser_active = False
hand_positions = {}

# Function to perform actions
def perform_action(action):
    global last_action_time
    if time.time() - last_action_time > 0.3:
        if action == "left":
            pyautogui.press('left')
        elif action == "right":
            pyautogui.press('right')
        elif action == "start_slideshow":
            pyautogui.press('f5')
        elif action == "exit":
            pyautogui.press('esc')
        elif action == "scroll_up":
            pyautogui.scroll(3)
        elif action == "scroll_down":
            pyautogui.scroll(-3)
        elif action == "laser_on":
            pyautogui.mouseDown()
        elif action == "laser_off":
            pyautogui.mouseUp()
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
            middle_tip = landmarks[12]
            
            hand_list.append((wrist, index_tip, thumb_tip, middle_tip))
            
            # Store hand positions to track movement
            if hand_idx not in hand_positions:
                hand_positions[hand_idx] = wrist.x
            movement = wrist.x - hand_positions[hand_idx]
            
            # Swipe Left (Previous Slide)
            if movement > 0.15:
                perform_action("left")
                hand_positions[hand_idx] = wrist.x
            # Swipe Right (Next Slide)
            elif movement < -0.15:
                perform_action("right")
                hand_positions[hand_idx] = wrist.x
            
            # Swipe Up (Scroll Up)
            if index_tip.y < 0.2:
                perform_action("scroll_up")
            # Swipe Down (Scroll Down)
            elif index_tip.y > 0.8:
                perform_action("scroll_down")
            
            # Thumbs Up (Start Slideshow)
            if thumb_tip.y < wrist.y:
                perform_action("start_slideshow")
            
            # Circular Gesture for Laser Pointer
            if abs(index_tip.x - middle_tip.x) < 0.05 and abs(index_tip.y - middle_tip.y) < 0.05:
                if not laser_active:
                    perform_action("laser_on")
                    laser_active = True
            else:
                if laser_active:
                    perform_action("laser_off")
                    laser_active = False
            
        # Zoom In/Out Detection (Two Hands Required)
        if len(hand_list) == 2:
            wrist1, index1, thumb1, _ = hand_list[0]
            wrist2, index2, thumb2, _ = hand_list[1]
            
            distance = np.linalg.norm(np.array([index1.x, index1.y]) - np.array([index2.x, index2.y]))
            
            if prev_distance is not None:
                if distance > prev_distance + 0.02:
                    pyautogui.hotkey('ctrl', '+')  # Zoom In
                elif distance < prev_distance - 0.02:
                    pyautogui.hotkey('ctrl', '-')  # Zoom Out
            
            prev_distance = distance
    
    cv2.imshow("Gesture Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
