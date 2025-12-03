import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


MAX_BUFFER = 1024            
DRAW_THICKNESS = 8
ERASE_THICKNESS = 40
SAVE_FILENAME_PREFIX = "air_writing_"


def lm_to_point(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def fingers_up_status(hand_landmarks, w, h):
    """
    Very simple finger-up test for the four fingers (thumb handled roughly).
    Returns list of booleans [thumb, index, middle, ring, pinky]
    We compare tip y with pip y (note: smaller y is higher on image).
    """
    lm = hand_landmarks.landmark
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids  = [2, 6, 10, 14, 18]  
    status = []
    for tip, pip in zip(tips_ids, pip_ids):
        status.append(lm[tip].y < lm[pip].y)
    return status

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    pts = deque(maxlen=MAX_BUFFER)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        last_time = time.time()
        save_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            drawing = False
            erasing = False

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                status = fingers_up_status(hand_landmarks, w, h)
                index_up = status[1]
                middle_up = status[2]

                lm = hand_landmarks.landmark
                ix, iy = lm_to_point(lm[8], w, h)

                if index_up and not middle_up:
                    drawing = True
                    pts.appendleft((ix, iy))
                elif index_up and middle_up:
                    erasing = True
                    cv2.circle(canvas, (ix, iy), ERASE_THICKNESS//2, (0,0,0), -1)
                else:
                    pts.appendleft(None)

            pts_list = list(pts)
            for i in range(1, len(pts_list)):
                if pts_list[i-1] is None or pts_list[i] is None:
                    continue
                cv2.line(canvas, pts_list[i-1], pts_list[i], (255, 255, 255), DRAW_THICKNESS, lineType=cv2.LINE_AA)

            overlay = frame.copy()
           
            canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(overlay, overlay, mask=mask_inv)
            strokes = np.zeros_like(overlay)
            strokes[:, :] = (255, 255, 255)
            strokes = cv2.bitwise_and(strokes, strokes, mask=mask)
            out = cv2.add(bg, strokes)

            
            mode_text = "Idle"
            if drawing:
                mode_text = "Drawing"
            elif erasing:
                mode_text = "Erasing"

            cv2.putText(out, f"Mode: {mode_text}  |  c:clear  s:save  q:quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Air Writing", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas[:] = 0
                pts.clear()
            elif key == ord('s'):
                fname = f"{SAVE_FILENAME_PREFIX}{int(time.time())}_{save_count}.png"
                cv2.imwrite(fname, canvas)
                print(f"Saved {fname}")
                save_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 