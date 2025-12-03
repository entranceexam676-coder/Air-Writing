import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MAX_BUFFER = 1024
DRAW_THICKNESS = 8
ERASE_THICKNESS = 60
SAVE_FILENAME_PREFIX = "air_writing_"

# Colors palette (BGR)
colors = [
    (255, 255, 255),  # White
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
]
color_names = ["White", "Green", "Red", "Blue", "Yellow", "Magenta"]

# UI layout (menu on the right)
MENU_WIDTH = 220
MENU_PADDING = 12
SWATCH_SIZE = 36
SWATCH_GAP = 12
BUTTON_H = 40
BUTTON_GAP = 12

window_name = "Air Writing"

def lm_to_point(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def fingers_up_status(hand_landmarks):
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

class Menu:
    def __init__(self, frame_w, frame_h):
        self.w = frame_w
        self.h = frame_h
        self.menu_x = frame_w - MENU_WIDTH
        self.menu_y = 0
        self.selected_color = 0
        self.tool = "pen"  # "pen" or "eraser"
        # compute menu items positions (dynamic layout)
        self._compute_layout()

    def _compute_layout(self):
        x0 = self.menu_x + MENU_PADDING
        y = MENU_PADDING + 40  # top offset for title
        # color swatches top block
        self.swatches = []
        for i in range(len(colors)):
            rx = x0
            ry = y + i * (SWATCH_SIZE + SWATCH_GAP)
            self.swatches.append((rx, ry, SWATCH_SIZE, SWATCH_SIZE))
        # Buttons after swatches
        y_buttons = y + len(colors) * (SWATCH_SIZE + SWATCH_GAP) + 16
        self.btn_pen = (x0, y_buttons, MENU_WIDTH - 2 * MENU_PADDING, BUTTON_H)
        self.btn_eraser = (x0, y_buttons + BUTTON_H + BUTTON_GAP, MENU_WIDTH - 2 * MENU_PADDING, BUTTON_H)
        self.btn_clear = (x0, y_buttons + 2 * (BUTTON_H + BUTTON_GAP), MENU_WIDTH - 2 * MENU_PADDING, BUTTON_H)
        self.btn_save = (x0, y_buttons + 3 * (BUTTON_H + BUTTON_GAP), MENU_WIDTH - 2 * MENU_PADDING, BUTTON_H)

    def draw(self, frame):
        # semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.menu_x, 0), (self.w, self.h), (8, 12, 20), -1)
        alpha = 0.65
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # title
        cv2.putText(frame, "Menu", (self.menu_x + MENU_PADDING, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)

        # color swatches and labels
        for i, (rx, ry, rw, rh) in enumerate(self.swatches):
            color = colors[i]
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, -1)
            # border highlight if selected
            if i == self.selected_color:
                cv2.rectangle(frame, (rx - 3, ry - 3), (rx + rw + 3, ry + rh + 3), (200, 200, 0), 2)
            # small name text right to swatch
            cv2.putText(frame, color_names[i], (rx + rw + 8, ry + rw // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        # buttons (pen / eraser / clear / save)
        self._draw_button(frame, self.btn_pen, "Pen" + (" ✓" if self.tool == "pen" else ""))
        self._draw_button(frame, self.btn_eraser, "Eraser" + (" ✓" if self.tool == "eraser" else ""))
        self._draw_button(frame, self.btn_clear, "Clear")
        self._draw_button(frame, self.btn_save, "Save")

        # current status at bottom
        status_text = f"Tool: {self.tool.title()}  Color: {color_names[self.selected_color]}"
        cv2.putText(frame, status_text, (self.menu_x + MENU_PADDING, self.h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

    def _draw_button(self, frame, rect, label):
        x, y, w, h = rect
        # button background
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 46, 60), -1)
        # border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 85, 100), 1)
        # text
        cv2.putText(frame, label, (x + 10, y + h//2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)

    def item_at(self, mx, my):
        """Return (type, index_or_name, rect) for the menu item under (mx,my) or None."""
        # swatches
        for i, (rx, ry, rw, rh) in enumerate(self.swatches):
            if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                return ("color", i, (rx, ry, rw, rh))
        # buttons
        for name, rect in [("pen", self.btn_pen), ("eraser", self.btn_eraser),
                           ("clear", self.btn_clear), ("save", self.btn_save)]:
            x, y, w, h = rect
            if x <= mx <= x + w and y <= my <= y + h:
                return ("button", name, rect)
        return None

    def handle_action(self, item_type, value, state):
        """Apply the menu action to the shared state."""
        if item_type == "color":
            self.selected_color = value
            state['selected_color'] = value
        elif item_type == "button":
            name = value
            if name == "pen":
                self.tool = "pen"
                state['tool'] = "pen"
            elif name == "eraser":
                self.tool = "eraser"
                state['tool'] = "eraser"
            elif name == "clear":
                state['clear_request'] = True
            elif name == "save":
                state['save_request'] = True

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def normalized_distance(lm1, lm2, frame_w, frame_h):
    # compute pixel distance then normalize relative to diagonal
    (x1, y1) = lm_to_point(lm1, frame_w, frame_h)
    (x2, y2) = lm_to_point(lm2, frame_w, frame_h)
    return math.hypot(x1-x2, y1-y2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pts = deque(maxlen=MAX_BUFFER)
    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # UI menu
    menu = Menu(frame_w, frame_h)

    # UI/state shared
    state = {
        'clear_request': False,
        'save_request': False,
        'selected_color': menu.selected_color,
        'tool': menu.tool,
    }

    # gesture menu interaction state
    dwell_start = None
    dwell_item = None
    DWELL_TIME = 0.8  # seconds to trigger hover-and-hold
    pinch_start = None
    PINCH_THRESH = min(frame_w, frame_h) * 0.05  # pixel threshold for pinch (about 5% of min dimension)
    last_activation_time = 0
    ACTIVATION_COOLDOWN = 0.5  # seconds to avoid double activations

    cv2.namedWindow(window_name)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
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

            index_point = None
            thumb_point = None
            pinch_distance = None
            is_pinching = False

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                status = fingers_up_status(hand_landmarks)
                index_up = status[1]
                middle_up = status[2]

                lm = hand_landmarks.landmark
                ix, iy = lm_to_point(lm[8], frame_w, frame_h)
                index_point = (ix, iy)
                tx, ty = lm_to_point(lm[4], frame_w, frame_h)
                thumb_point = (tx, ty)

                pinch_distance = distance(index_point, thumb_point)
                if pinch_distance < PINCH_THRESH:
                    is_pinching = True
                else:
                    is_pinching = False

                # Gesture-driven drawing vs erasing (same as before)
                # If both index+middle -> explicit erase circle
                if index_up and middle_up:
                    erasing = True
                    cv2.circle(canvas, (ix, iy), ERASE_THICKNESS // 2, (0, 0, 0), -1)
                    pts.appendleft((ix, iy))
                elif index_up and not middle_up:
                    if state['tool'] == "pen":
                        drawing = True
                        pts.appendleft((ix, iy))
                    else:
                        erasing = True
                        cv2.circle(canvas, (ix, iy), ERASE_THICKNESS // 2, (0, 0, 0), -1)
                        pts.appendleft((ix, iy))
                else:
                    pts.appendleft(None)

            # Update menu from state (in case recently changed)
            menu.selected_color = state['selected_color']
            menu.tool = state['tool']

            # Draw lines from pts buffer to canvas (use pen color or eraser)
            pen_color = colors[menu.selected_color]
            pts_list = list(pts)
            for i in range(1, len(pts_list)):
                if pts_list[i-1] is None or pts_list[i] is None:
                    continue
                p1 = pts_list[i-1]
                p2 = pts_list[i]
                if menu.tool == "eraser":
                    cv2.line(canvas, p1, p2, (0, 0, 0), ERASE_THICKNESS, lineType=cv2.LINE_AA)
                else:
                    cv2.line(canvas, p1, p2, pen_color, DRAW_THICKNESS, lineType=cv2.LINE_AA)

            # Build final display by overlaying strokes where canvas has content
            overlay = frame.copy()
            canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(overlay, overlay, mask=mask_inv)
            strokes = cv2.bitwise_and(canvas, canvas, mask=mask)
            out = cv2.add(bg, strokes)

            # Show cursor
            if index_point is not None:
                cx, cy = index_point
                # cursor circle
                if menu.tool == "eraser":
                    cv2.circle(out, (cx, cy), ERASE_THICKNESS // 2, (200, 200, 200), 2)
                else:
                    cv2.circle(out, (cx, cy), DRAW_THICKNESS // 2 + 2, pen_color, 2)

            # Gesture-based menu interaction
            activated = False
            now = time.time()
            if index_point is not None:
                # check if over a menu item
                item = menu.item_at(index_point[0], index_point[1])
                if item is not None:
                    item_type, item_val, rect = item
                    # draw highlight rectangle / ring
                    rx, ry, rw, rh = rect
                    # lighten the item visually
                    cv2.rectangle(out, (rx-2, ry-2), (rx+rw+2, ry+rh+2), (255,255,255), 1)

                    # If pinching over the item -> immediate activation (with cooldown)
                    if is_pinching and (now - last_activation_time) > ACTIVATION_COOLDOWN:
                        menu.handle_action(item_type if item_type=="color" else "button", item_val, state)
                        last_activation_time = now
                        activated = True
                        # flash visual feedback: fill small rectangle
                        if item_type == "color":
                            cv2.rectangle(out, (rx, ry), (rx+rw, ry+rh), colors[item_val], -1)
                            cv2.rectangle(out, (rx-2, ry-2), (rx+rw+2, ry+rh+2), (0,255,0), 2)
                        else:
                            x, y, w, h = rect
                            cv2.rectangle(out, (x, y), (x+w, y+h), (0,255,0), -1)
                        # reset dwell
                        dwell_start = None
                        dwell_item = None
                    else:
                        # Hover dwell: if pointing (index up) and steady over same item for DWELL_TIME -> activate
                        if dwell_item is None or dwell_item != (item_type, item_val):
                            dwell_item = (item_type, item_val)
                            dwell_start = now
                        else:
                            elapsed = now - dwell_start if dwell_start else 0
                            # draw dwell progress arc (simple circular progress near cursor)
                            progress = min(1.0, elapsed / DWELL_TIME)
                            # small progress circle around cursor
                            if index_point is not None:
                                cx, cy = index_point
                                radius = 22
                                # background ring
                                cv2.circle(out, (cx, cy), radius, (80,80,80), 2)
                                # progress arc by drawing multiple small lines
                                end_angle = int(360 * progress)
                                # approximate arc by drawing filled pie slice using ellipse
                                cv2.ellipse(out, (cx, cy), (radius, radius), -90, 0, end_angle, (0,200,0), 4)
                            if elapsed >= DWELL_TIME and (now - last_activation_time) > ACTIVATION_COOLDOWN:
                                menu.handle_action(item_type if item_type=="color" else "button", item_val, state)
                                last_activation_time = now
                                activated = True
                                dwell_start = None
                                dwell_item = None
                else:
                    # not pointing at any menu item: reset dwell
                    dwell_start = None
                    dwell_item = None
            else:
                dwell_start = None
                dwell_item = None

            # draw menu on the right
            menu.draw(out)

            # on-screen hints
            cv2.putText(out, "Gestures: Index = draw/erase | Index+Middle = erase", (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(out, "To operate menu: Point + Pinch (quick) OR Point + Hold (0.8s)", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, out)

            # handle menu requests
            if state['clear_request']:
                canvas[:] = 0
                pts.clear()
                state['clear_request'] = False

            if state['save_request']:
                fname = f"{SAVE_FILENAME_PREFIX}{int(time.time())}_{save_count}.png"
                cv2.imwrite(fname, canvas)
                print(f"Saved {fname}")
                save_count += 1
                state['save_request'] = False

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
            elif key == ord('e'):
                state['tool'] = "eraser"
            elif key == ord('p'):
                state['tool'] = "pen"
            elif key == ord('n'):
                state['selected_color'] = (state['selected_color'] + 1) % len(colors)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
