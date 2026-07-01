import cv2
import numpy as np
import math
import pyautogui

LOWER_SKIN_HSV = np.array([0, 15, 45], dtype=np.uint8)
UPPER_SKIN_HSV = np.array([30, 255, 190], dtype=np.uint8)
LOWER_SKIN_YCRCB = np.array([0, 133, 77], dtype=np.uint8)
UPPER_SKIN_YCRCB = np.array([255, 173, 127], dtype=np.uint8)


MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


SKIN_MASK_DOWNSCALE = 0.6
BLUR_KERNEL = (7, 7)
MORPH_ITERATIONS = 2
CENTER_HISTORY_LENGTH = 5


MOTION_MIN_AREA = 1200
MOTION_THRESHOLD = 8.0 
STATIONARY_FRAMES = 4
MOUSE_SMOOTHING = 0.45
MOVE_DEADZONE = 4.0


def get_skin_mask(frame,
                  lower_hsv=LOWER_SKIN_HSV,
                  upper_hsv=UPPER_SKIN_HSV,
                  lower_ycrcb=LOWER_SKIN_YCRCB,
                  upper_ycrcb=UPPER_SKIN_YCRCB):
    """Return a cleaned binary mask of skin-colored regions."""
    scale = SKIN_MASK_DOWNSCALE
    if scale < 1.0:
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        small = frame

    blurred = cv2.GaussianBlur(small, BLUR_KERNEL, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=MORPH_ITERATIONS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=MORPH_ITERATIONS)
    mask = cv2.medianBlur(mask, 5)

    if scale < 1.0:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask


def find_largest_contour_center(mask):
    """Find the centroid of the largest contour in the mask, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MOTION_MIN_AREA:
        return None

    hull = cv2.convexHull(largest)
    M = cv2.moments(hull)
    if M.get('m00', 0) == 0:
        return None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy, hull


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # frame and screen sizes
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    screen_width, screen_height = pyautogui.size()

    prev_center = None
    filtered_center = None
    center_history = []
    stationary_count = 0
    last_mouse_x = None
    last_mouse_y = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        mask = get_skin_mask(frame)
        result = find_largest_contour_center(mask)

        motion_text = "No hand detected"
        motion = False
        if result:
            cx, cy, contour = result
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

            # map camera coords to screen coords and smooth mouse movement
            raw_center = np.array([cx, cy], dtype=float)
            center_history.append(raw_center)
            if len(center_history) > CENTER_HISTORY_LENGTH:
                center_history.pop(0)

            median_center = np.median(center_history, axis=0)
            if filtered_center is None:
                filtered_center = median_center
            else:
                filtered_center += (median_center - filtered_center) * MOUSE_SMOOTHING

            screen_x = (filtered_center[0] / frame_width) * screen_width
            screen_y = (filtered_center[1] / frame_height) * screen_height

            if last_mouse_x is None:
                last_mouse_x = screen_x
                last_mouse_y = screen_y
                motion = True
                motion_text = "Tracking..."

            smooth_x = last_mouse_x + (screen_x - last_mouse_x) * MOUSE_SMOOTHING
            smooth_y = last_mouse_y + (screen_y - last_mouse_y) * MOUSE_SMOOTHING

            if prev_center is not None:
                dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
                if dist > MOTION_THRESHOLD:
                    motion = True
                    stationary_count = 0
                    motion_text = "Motion Detected"
                else:
                    stationary_count += 1
                    if stationary_count >= STATIONARY_FRAMES:
                        motion_text = "Stationary"
                    else:
                        motion_text = "Almost stationary"
            else:
                # Keep the first detected position, but don't treat it as continued motion.
                motion_text = "Tracking..."

            if motion and math.hypot(smooth_x - last_mouse_x, smooth_y - last_mouse_y) > MOVE_DEADZONE:
                try:
                    pyautogui.moveTo(min(max(int(smooth_x), 0), screen_width - 1),
                                     min(max(int(smooth_y), 0), screen_height - 1))
                    last_mouse_x = smooth_x
                    last_mouse_y = smooth_y
                except Exception:
                    pass

            prev_center = (cx, cy)

        cv2.putText(frame, motion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow('Motion Only - Hand Detector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            try:
                cv2.imshow('Mask', mask)
            except Exception:
                pass
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
 