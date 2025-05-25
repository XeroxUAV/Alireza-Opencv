import cv2
import numpy as np
from djitellopy import Tello
import time

# PID controller
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0
        self.last_time = None

    def update(self, error):
        current_time = time.time()
        dt = 0.03 if self.last_time is None else max(current_time - self.last_time, 0.03)
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        self.last_time = current_time
        return output

# Kalman filter
class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.zeros((4, 1), np.float32)
        self.kalman.statePost = np.zeros((4, 1), np.float32)

    def predict(self):
        return self.kalman.predict()

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)

# Tello setup
print("Connecting to Tello...")
tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()
tello.send_rc_control(0, 0, 0, 0)
time.sleep(0.5)
print("Battery:", tello.get_battery())

# HSV trackbars
cv2.namedWindow("HSV")
for name, val in zip(["LH", "LS", "LV", "UH", "US", "UV", "MinArea", "Kernel", "Iteration"], [110, 72, 42, 179, 255, 255, 11042, 5, 0]):
    cv2.createTrackbar(name, "HSV", val, 255 if 'H' in name or 'S' in name or 'V' in name else 50 if name == "Kernel" else 50000 if name == "MinArea" else 10, lambda x: None)

pid_yaw = PID(Kp=0.2, Ki=0.001, Kd=0.1)
kf = KalmanFilter()
alpha = 0.2
smoothed_center = None
last_confidence = 1.0
center_x = 320

# Main loop
while True:
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV and mask
    lh, ls, lv = [cv2.getTrackbarPos(f"L{c}", "HSV") for c in "HSV"]
    uh, us, uv = [cv2.getTrackbarPos(f"U{c}", "HSV") for c in "HSV"]
    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])
    mask = cv2.inRange(hsv, lower, upper)

    kernel_size = max(1, cv2.getTrackbarPos("Kernel", "HSV"))
    min_area = cv2.getTrackbarPos("MinArea", "HSV")
    iteration = cv2.getTrackbarPos("Iteration", "HSV")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iteration)
    dilated = cv2.dilate(mask, kernel, iterations=iteration)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_window = max((cnt for cnt in contours if 4 <= len(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) <= 6 and cv2.contourArea(cnt) > min_area), key=cv2.contourArea, default=None)

    if best_window is not None:
        M = cv2.moments(best_window)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            smoothed_center = (int(alpha * cX + (1 - alpha) * smoothed_center[0]), int(alpha * cY + (1 - alpha) * smoothed_center[1])) if smoothed_center else (cX, cY)
            kf.correct(*smoothed_center)
            error_x = smoothed_center[0] - center_x

            yaw_speed = np.sign(error_x) * 10 if abs(error_x) > 100 else int(np.clip(pid_yaw.update(error_x), -100, 100))
            tello.send_rc_control(0, 0, 0, yaw_speed)
            last_confidence = 1.0
        else:
            tello.send_rc_control(0, 0, 0, 0)
    else:
        last_confidence *= 0.95
        tello.send_rc_control(0, 0, 0, 0)

    predicted = tuple(map(int, kf.predict()[:2]))
    if smoothed_center:
        cv2.circle(frame, smoothed_center, 5, (0, 0, 255), -1)
        cv2.line(frame, (center_x, 240), smoothed_center, (255, 0, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw_speed}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.circle(frame, predicted, 5, (255, 0, 0), -1)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
        break

cv2.destroyAllWindows()
tello.streamoff()