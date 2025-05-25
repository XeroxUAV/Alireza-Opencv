import cv2
import numpy as np
from djitellopy import Tello
import time




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
        if self.last_time is None:
            dt = 0.03
        else:
            dt = current_time - self.last_time
            if dt <= 0:
                dt = 0.03

        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.previous_error = error
        self.last_time = current_time
        return output


class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.zeros((4, 1), np.float32)
        self.kalman.statePost = np.zeros((4, 1), np.float32)

    def predict(self):
        prediction = self.kalman.predict()
        return prediction

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)


class WindowDetector:
    def __init__(self):
        self.kf = KalmanFilter()
        self.alpha = 0.2
        self.smoothed_center = None
        self.last_confidence = 1.0
        self.decay_factor = 0.95

        cv2.namedWindow("HSV")
        # HSV1
        cv2.createTrackbar("LH1", "HSV", 114, 179, lambda x: None)
        cv2.createTrackbar("LS1", "HSV", 89, 255, lambda x: None)
        cv2.createTrackbar("LV1", "HSV", 145, 255, lambda x: None)
        cv2.createTrackbar("UH1", "HSV", 179, 179, lambda x: None)
        cv2.createTrackbar("US1", "HSV", 255, 255, lambda x: None)
        cv2.createTrackbar("UV1", "HSV", 255, 255, lambda x: None)
        # HSV2
        cv2.createTrackbar("LH2", "HSV", 114, 179, lambda x: None)
        cv2.createTrackbar("LS2", "HSV", 103, 255, lambda x: None)
        cv2.createTrackbar("LV2", "HSV", 129, 255, lambda x: None)
        cv2.createTrackbar("UH2", "HSV", 128, 179, lambda x: None)
        cv2.createTrackbar("US2", "HSV", 255, 255, lambda x: None)
        cv2.createTrackbar("UV2", "HSV", 255, 255, lambda x: None)
        # HSV3
        cv2.createTrackbar("LH3", "HSV", 116, 179, lambda x: None)
        cv2.createTrackbar("LS3", "HSV", 36, 255, lambda x: None)
        cv2.createTrackbar("LV3", "HSV", 153, 255, lambda x: None)
        cv2.createTrackbar("UH3", "HSV", 145, 179, lambda x: None)
        cv2.createTrackbar("US3", "HSV", 128, 255, lambda x: None)
        cv2.createTrackbar("UV3", "HSV", 255, 255, lambda x: None)

        cv2.createTrackbar("Iteration", "HSV", 2, 10, lambda x: None)
        cv2.createTrackbar("MinArea", "HSV", 7500, 50000, lambda x: None)
        cv2.createTrackbar("Kernel", "HSV", 3, 50, lambda x: None)

    def get_hsv_ranges(self):
        # Read HSV1
        lh1 = cv2.getTrackbarPos("LH1", "HSV")
        ls1 = cv2.getTrackbarPos("LS1", "HSV")
        lv1 = cv2.getTrackbarPos("LV1", "HSV")
        uh1 = cv2.getTrackbarPos("UH1", "HSV")
        us1 = cv2.getTrackbarPos("US1", "HSV")
        uv1 = cv2.getTrackbarPos("UV1", "HSV")
        lower1 = np.array([lh1, ls1, lv1])
        upper1 = np.array([uh1, us1, uv1])
        # Read HSV2
        lh2 = cv2.getTrackbarPos("LH2", "HSV")
        ls2 = cv2.getTrackbarPos("LS2", "HSV")
        lv2 = cv2.getTrackbarPos("LV2", "HSV")
        uh2 = cv2.getTrackbarPos("UH2", "HSV")
        us2 = cv2.getTrackbarPos("US2", "HSV")
        uv2 = cv2.getTrackbarPos("UV2", "HSV")
        lower2 = np.array([lh2, ls2, lv2])
        upper2 = np.array([uh2, us2, uv2])
        # Read HSV3
        lh3 = cv2.getTrackbarPos("LH3", "HSV")
        ls3 = cv2.getTrackbarPos("LS3", "HSV")
        lv3 = cv2.getTrackbarPos("LV3", "HSV")
        uh3 = cv2.getTrackbarPos("UH3", "HSV")
        us3 = cv2.getTrackbarPos("US3", "HSV")
        uv3 = cv2.getTrackbarPos("UV3", "HSV")
        lower3 = np.array([lh3, ls3, lv3])
        upper3 = np.array([uh3, us3, uv3])

        return (lower1, upper1, lower2, upper2, lower3, upper3)

def process(self, frame, pid_yaw, tello, pid_throttle=None):
    frame_h, frame_w = frame.shape[:2]
    frame_center = (frame_w // 2, frame_h // 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1, upper1, lower2, upper2, lower3, upper3 = self.get_hsv_ranges()

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask3 = cv2.inRange(hsv, lower3, upper3)
    combined_mask = cv2.bitwise_or(mask1, mask2)
    combined_mask = cv2.bitwise_or(combined_mask, mask3)

    kernelTrack = cv2.getTrackbarPos("Kernel", "HSV")
    min_area = cv2.getTrackbarPos("MinArea", "HSV")
    iteration = cv2.getTrackbarPos("Iteration", "HSV")

    kernel = np.ones((max(1, kernelTrack), max(1, kernelTrack)), np.uint8)
    mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=iteration)
    dilated = cv2.dilate(mask_cleaned, kernel, iterations=iteration)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_window = None
    best_area = 0

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(approx)
        if 4 <= len(approx) <= 6 and area > min_area:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 3.0 and area > best_area:
                best_window = approx
                best_area = area

    if best_window is not None:
        M = cv2.moments(best_window)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if self.smoothed_center is None:
                self.smoothed_center = (cX, cY)
                initial_state = np.array([[np.float32(cX)], [np.float32(cY)], [0], [0]], dtype=np.float32)
                self.kf.kalman.statePre = initial_state.copy()
                self.kf.kalman.statePost = initial_state.copy()
            else:
                self.smoothed_center = (
                    int(self.alpha * cX + (1 - self.alpha) * self.smoothed_center[0]),
                    int(self.alpha * cY + (1 - self.alpha) * self.smoothed_center[1])
                )
                self.kf.correct(self.smoothed_center[0], self.smoothed_center[1])

            # جبران زاویه دوربین با offset
            adjusted_center_y = frame_center[1] - 60

            error_x = self.smoothed_center[0] - frame_center[0]
            error_y = self.smoothed_center[1] - adjusted_center_y

            # کنترل Yaw
            error_threshold = 100
            min_yaw_speed = 10
            if abs(error_x) > error_threshold:
                yaw_speed = int(np.sign(error_x) * min_yaw_speed)
            else:
                yaw_speed = int(np.clip(pid_yaw.update(error_x), -100, 100))

            # کنترل Throttle
            throttle = 0
            if pid_throttle is not None:
                throttle = int(np.clip(pid_throttle.update(-error_y), -50, 50))  # منفی چون پایین رفتن یعنی y بیشتر

            # ارسال فرمان
            tello.send_rc_control(0, 0, throttle, yaw_speed)

            # نمایش
            cv2.drawContours(frame, [best_window], 0, (0, 255, 0), 3)
            cv2.circle(frame, self.smoothed_center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Center: {self.smoothed_center}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Yaw speed: {yaw_speed}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Throttle: {throttle}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Error X: {error_x}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Error Y: {error_y}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.last_confidence = 1.0
        else:
            tello.send_rc_control(0, 0, 0, 0)
    else:
        self.last_confidence *= self.decay_factor
        cv2.putText(frame, f"Confidence: {self.last_confidence:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tello.send_rc_control(0, 0, 0, 0)

    predicted = self.kf.predict()
    predicted = (int(predicted[0]), int(predicted[1]))
    cv2.circle(frame, predicted, 5, (255, 0, 0), -1)
    cv2.putText(frame, f"Predicted: {predicted}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Combined Mask", combined_mask)
    cv2.imshow("Mask1", mask1)
    cv2.imshow("Mask2", mask2)
    cv2.imshow("Mask3", mask3)

    return frame

class DroneController:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.tello.takeoff()
        # Immediately climb a bit more
        self.tello.send_rc_control(0, 0, 30, 0)  # 30 = moderate throttle up
        time.sleep(5)  # Let it climb for 5 second
        self.tello.send_rc_control(0, 0, 0, 0)  # Stop vertical movement


        time.sleep(0.5)  # یک مکث کوتاه تا دستور به‌درستی اعمال بشه
        self.pid_yaw = PID(Kp=0.2, Ki=0.01, Kd=0.1)
        self.pid_throttle = PID(Kp=0.3, Ki=0.01, Kd=0.1)

        self.detector = WindowDetector()

    def run(self):
        while True:
            frame = self.tello.get_frame_read().frame
            frame = cv2.resize(frame, (640, 480))

            output = self.detector.process(frame, self.pid_yaw, self.tello, self.pid_throttle)
            cv2.imshow("Window Detection", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.tello.send_rc_control(0, 0, 0, 0)
                self.tello.land()
                break

        cv2.destroyAllWindows()
        self.tello.streamoff()


if __name__ == "__main__":
    controller = DroneController()
    controller.run()