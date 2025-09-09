import cv2
import time
import math
import numpy as np
import pyautogui
import argparse
import os

try:
    import mediapipe as mp
except Exception as exc:
    raise SystemExit("mediapipe is required. Install with: pip install mediapipe") from exc

# Optional: Windows system volume control via pycaw
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    HAS_PYCAW = True
except Exception:
    HAS_PYCAW = False

# Optional: brightness control
try:
    import screen_brightness_control as sbc
    HAS_BRIGHTNESS = True
except Exception:
    HAS_BRIGHTNESS = False


# Configuration
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()
CAM_W, CAM_H = 960, 720
SMOOTHING = 0.25  # lower is smoother, 0..1
CLICK_DIST_PX = 40  # pinch threshold for clicks (scaled to camera width)
DRAG_DIST_PX = 35  # pinch threshold for drag
SCROLL_SENSITIVITY = 0.7
VOL_SENSITIVITY = 0.6
BRIGHT_SENSITIVITY = 0.6
ZOOM_MIN_DIST = 20
ZOOM_MAX_DIST = 220


# Helpers for volume control
class SystemVolumeController:
    def __init__(self):
        self.available = False
        self.endpoint = None
        self.volume = None
        if HAS_PYCAW:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume = cast(interface, POINTER(IAudioEndpointVolume))
                self.available = True
            except Exception:
                self.available = False

    def get_volume(self) -> float:
        if not self.available:
            return 0.0
        try:
            return self.volume.GetMasterVolumeLevelScalar()
        except Exception:
            return 0.0

    def set_volume(self, scalar: float) -> None:
        if not self.available:
            return
        scalar = max(0.0, min(1.0, scalar))
        try:
            self.volume.SetMasterVolumeLevelScalar(scalar, None)
        except Exception:
            pass


# Brightness control wrapper
class BrightnessController:
    def __init__(self):
        self.available = HAS_BRIGHTNESS

    def get_brightness(self) -> int:
        if not self.available:
            return 50
        try:
            vals = sbc.get_brightness()
            if isinstance(vals, list) and vals:
                return int(vals[0])
            return int(vals)
        except Exception:
            return 50

    def set_brightness(self, value: int) -> None:
        if not self.available:
            return
        try:
            value = int(max(0, min(100, value)))
            sbc.set_brightness(value)
        except Exception:
            pass


# Hand utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

FINGER_TIPS = [4, 8, 12, 16, 20]


def fingers_up(landmarks, handedness: str):
    finger_states = [False] * 5
    if handedness == 'Right':
        finger_states[0] = landmarks[4].x < landmarks[3].x
    else:
        finger_states[0] = landmarks[4].x > landmarks[3].x
    finger_states[1] = landmarks[8].y < landmarks[6].y
    finger_states[2] = landmarks[12].y < landmarks[10].y
    finger_states[3] = landmarks[16].y < landmarks[14].y
    finger_states[4] = landmarks[20].y < landmarks[18].y
    return finger_states


def landmark_to_px(lm, width, height):
    return int(lm.x * width), int(lm.y * height)


def distance(a, b, width, height):
    ax, ay = landmark_to_px(a, width, height)
    bx, by = landmark_to_px(b, width, height)
    return math.hypot(ax - bx, ay - by)


def lerp(a, b, t):
    return a + (b - a) * t


# Image manipulation (zoom/rotate/flip/pan)
class DemoImage:
    def __init__(self, image_path: str | None = None, target_size=(480, 320)):
        self.base = self._load_image(image_path)
        self.zoom = 1.0
        self.angle = 0.0
        self.flip_h = False
        self.last_zoom_ref = None
        self.last_angle_ref = None
        self.offset_x = 0
        self.offset_y = 0
        self.target_w, self.target_h = target_size

    def _load_image(self, image_path: str | None):
        if image_path and os.path.isfile(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                return img
        # fallback gradient
        w, h = 800, 600
        x = np.linspace(0, 255, w, dtype=np.uint8)
        y = np.linspace(0, 255, h, dtype=np.uint8)
        xv, yv = np.meshgrid(x, y)
        img = np.stack([xv, yv, ((xv.astype(np.int32) + yv.astype(np.int32)) // 2).astype(np.uint8)], axis=2)
        return img

    def toggle_flip(self):
        self.flip_h = not self.flip_h

    def set_zoom_from_distance(self, d: float):
        d = max(ZOOM_MIN_DIST, min(ZOOM_MAX_DIST, d))
        t = (d - ZOOM_MIN_DIST) / (ZOOM_MAX_DIST - ZOOM_MIN_DIST)
        self.zoom = 0.5 + 1.5 * t

    def adjust_zoom(self, delta_scale: float):
        self.zoom = float(max(0.2, min(4.0, self.zoom * (1.0 + delta_scale))))

    def set_angle_from_vector(self, dx: float, dy: float):
        self.angle = math.degrees(math.atan2(dy, dx))

    def pan_by(self, dx: int, dy: int):
        self.offset_x += int(dx)
        self.offset_y += int(dy)

    def render(self):
        img = self.base.copy()
        if self.flip_h:
            img = cv2.flip(img, 1)
        # scale
        if not math.isclose(self.zoom, 1.0, rel_tol=1e-3):
            new_w = int(img.shape[1] * self.zoom)
            new_h = int(img.shape[0] * self.zoom)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # rotate
        if not math.isclose(self.angle, 0.0, abs_tol=0.5):
            center = (img.shape[1] // 2, img.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        # canvas
        target_w, target_h = self.target_w, self.target_h
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        # pan
        x0 = (target_w - img.shape[1]) // 2 + self.offset_x
        y0 = (target_h - img.shape[0]) // 2 + self.offset_y
        x1 = x0 + img.shape[1]
        y1 = y0 + img.shape[0]
        # compute overlap
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = min(img.shape[1], target_w - x0)
        sy1 = min(img.shape[0], target_h - y0)
        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = min(target_w, x1)
        dy1 = min(target_h, y1)
        if sx1 > sx0 and sy1 > sy0:
            canvas[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
        return canvas


class GestureMouse:
    def __init__(self, image_path: str | None = None):
        self.prev_mouse_x = None
        self.prev_mouse_y = None
        self.dragging = False
        self.last_left_click_time = 0.0
        self.last_right_click_time = 0.0
        self.scroll_anchor_y = None
        self.volume_anchor_y = None
        self.brightness_anchor_y = None
        self.initial_volume = None
        self.initial_brightness = None
        self.vol = SystemVolumeController()
        self.bright = BrightnessController()
        # Image demo/viewer
        self.image_mode = image_path is not None
        self.demo_image = DemoImage(image_path=image_path, target_size=(640, 400)) if self.image_mode else None
        self.prev_index_px = None  # for panning

    def move_cursor(self, x_cam, y_cam):
        target_x = np.interp(x_cam, [0, CAM_W], [0, SCREEN_W])
        target_y = np.interp(y_cam, [0, CAM_H], [0, SCREEN_H])
        if self.prev_mouse_x is None:
            smoothed_x, smoothed_y = target_x, target_y
        else:
            smoothed_x = lerp(self.prev_mouse_x, target_x, 1.0 - SMOOTHING)
            smoothed_y = lerp(self.prev_mouse_y, target_y, 1.0 - SMOOTHING)
        self.prev_mouse_x, self.prev_mouse_y = smoothed_x, smoothed_y
        if not self.image_mode:
            pyautogui.moveTo(smoothed_x, smoothed_y)

    def handle_left_click(self):
        if self.image_mode:
            return
        now = time.time()
        if now - self.last_left_click_time > 0.25:
            pyautogui.click(button='left')
            self.last_left_click_time = now

    def handle_right_click(self):
        if self.image_mode:
            return
        now = time.time()
        if now - self.last_right_click_time > 0.35:
            pyautogui.click(button='right')
            self.last_right_click_time = now

    def handle_drag(self, start: bool):
        if self.image_mode:
            return
        if start and not self.dragging:
            pyautogui.mouseDown()
            self.dragging = True
        elif not start and self.dragging:
            pyautogui.mouseUp()
            self.dragging = False

    def handle_scroll(self, delta_y):
        if self.image_mode:
            # map scroll to zoom when in image mode
            if self.demo_image is not None:
                self.demo_image.adjust_zoom(-delta_y * 0.002)
            return
        pyautogui.scroll(int(-delta_y * SCROLL_SENSITIVITY))

    def start_volume_mode(self, anchor_y: int):
        if self.image_mode:
            return
        self.volume_anchor_y = anchor_y
        self.initial_volume = self.vol.get_volume()

    def update_volume(self, current_y: int):
        if self.image_mode:
            return
        if self.volume_anchor_y is None or self.initial_volume is None:
            return
        dy = self.volume_anchor_y - current_y
        new_vol = self.initial_volume + (dy / CAM_H) * VOL_SENSITIVITY
        self.vol.set_volume(new_vol)

    def end_volume_mode(self):
        self.volume_anchor_y = None
        self.initial_volume = None

    def start_brightness_mode(self, anchor_y: int):
        if self.image_mode:
            return
        self.brightness_anchor_y = anchor_y
        self.initial_brightness = self.bright.get_brightness()

    def update_brightness(self, current_y: int):
        if self.image_mode:
            return
        if self.brightness_anchor_y is None or self.initial_brightness is None:
            return
        dy = self.brightness_anchor_y - current_y
        new_b = int(self.initial_brightness + (dy / CAM_H) * 100 * BRIGHT_SENSITIVITY)
        self.bright.set_brightness(new_b)

    def end_brightness_mode(self):
        self.brightness_anchor_y = None
        self.initial_brightness = None


# Main loop

def main():
    parser = argparse.ArgumentParser(description="Virtual Mouse with Hand Gestures")
    parser.add_argument('--image', type=str, default=None, help='Path to an image to control (zoom/rotate/flip/pan).')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    gesture_mouse = GestureMouse(image_path=None)

    with mp_hands.Hands(static_image_mode=False,
                         max_num_hands=1,
                         model_complexity=1,
                         min_detection_confidence=0.6,
                         min_tracking_confidence=0.6) as hands:
        prev_time = 0
        scroll_prev_y = None
        flip_cooldown = 0.0

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            h, w, _ = frame.shape
            cam_w, cam_h = w, h

            # Only draw image panel if image mode is enabled
            panel_height = 0
            if gesture_mouse.image_mode and gesture_mouse.demo_image is not None:
                demo = gesture_mouse.demo_image.render()
                dh, dw = demo.shape[0], demo.shape[1]
                frame[0:dh, 0:dw] = demo
                panel_height = dh

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handed_label = results.multi_handedness[0].classification[0].label
                lm = hand_landmarks.landmark

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                f_up = fingers_up(lm, handed_label)
                thumb_up, index_up, middle_up, ring_up, pinky_up = f_up

                ix, iy = landmark_to_px(lm[8], cam_w, cam_h)
                mx, my = landmark_to_px(lm[12], cam_w, cam_h)

                # Cursor move: index only
                if index_up and not middle_up and not ring_up and not pinky_up:
                    gesture_mouse.move_cursor(ix, iy)

                # Pinches
                pinch_index_thumb = distance(lm[8], lm[4], cam_w, cam_h) < CLICK_DIST_PX
                pinch_middle_thumb = distance(lm[12], lm[4], cam_w, cam_h) < CLICK_DIST_PX

                if not gesture_mouse.image_mode:
                    if pinch_index_thumb and not middle_up:
                        gesture_mouse.handle_left_click()
                    if pinch_middle_thumb and index_up:
                        gesture_mouse.handle_right_click()

                # Drag or pan
                drag = distance(lm[8], lm[4], cam_w, cam_h) < DRAG_DIST_PX
                if gesture_mouse.image_mode and gesture_mouse.demo_image is not None:
                    if drag:
                        if gesture_mouse.prev_index_px is None:
                            gesture_mouse.prev_index_px = (ix, iy)
                        else:
                            dx = ix - gesture_mouse.prev_index_px[0]
                            dy = iy - gesture_mouse.prev_index_px[1]
                            gesture_mouse.demo_image.pan_by(dx, dy)
                            gesture_mouse.prev_index_px = (ix, iy)
                    else:
                        gesture_mouse.prev_index_px = None
                else:
                    gesture_mouse.handle_drag(drag)

                # Scroll/Zoom: index + middle up
                if index_up and middle_up and not ring_up and not pinky_up:
                    if scroll_prev_y is None:
                        scroll_prev_y = iy
                    else:
                        dy = iy - scroll_prev_y
                        gesture_mouse.handle_scroll(dy)
                        scroll_prev_y = iy
                else:
                    scroll_prev_y = None

                # Image zoom/rotate with index-thumb vector always updates panel
                if gesture_mouse.image_mode and gesture_mouse.demo_image is not None:
                    pinch_dist = distance(lm[8], lm[4], cam_w, cam_h)
                    gesture_mouse.demo_image.set_zoom_from_distance(pinch_dist)
                    dxv = lm[8].x - lm[4].x
                    dyv = lm[8].y - lm[4].y
                    gesture_mouse.demo_image.set_angle_from_vector(dxv, dyv)

                # Flip image when only pinky up
                now = time.time()
                if gesture_mouse.image_mode and gesture_mouse.demo_image is not None:
                    if pinky_up and not index_up and not middle_up and not ring_up:
                        if now - flip_cooldown > 0.8:
                            gesture_mouse.demo_image.toggle_flip()
                            flip_cooldown = now

                # Volume (ring only) if not in image mode
                if not gesture_mouse.image_mode:
                    if ring_up and not index_up and not middle_up and not pinky_up:
                        if gesture_mouse.volume_anchor_y is None:
                            gesture_mouse.start_volume_mode(iy)
                        else:
                            gesture_mouse.update_volume(iy)
                    else:
                        if gesture_mouse.volume_anchor_y is not None:
                            gesture_mouse.end_volume_mode()

                # Brightness (all up) if not in image mode
                if not gesture_mouse.image_mode:
                    if thumb_up and index_up and middle_up and ring_up and pinky_up:
                        if gesture_mouse.brightness_anchor_y is None:
                            gesture_mouse.start_brightness_mode(iy)
                        else:
                            gesture_mouse.update_brightness(iy)
                    else:
                        if gesture_mouse.brightness_anchor_y is not None:
                            gesture_mouse.end_brightness_mode()

                # UI text
                mode_txt = "IMAGE MODE" if gesture_mouse.image_mode else "SYSTEM MODE"
                cv2.putText(frame, mode_txt, (10, panel_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(frame, f"Fingers: T{int(thumb_up)} I{int(index_up)} M{int(middle_up)} R{int(ring_up)} P{int(pinky_up)}",
                            (10, cam_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (cam_w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow('Virtual Mouse', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    gesture_mouse.handle_drag(False)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

