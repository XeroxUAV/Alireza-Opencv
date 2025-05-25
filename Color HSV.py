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

        # ایجاد پنجره تنظیمات HSV و مقادیر مرتبط
        cv2.namedWindow("HSV")
        cv2.createTrackbar("LH1", "HSV", 110, 179, lambda x: None)
        cv2.createTrackbar("LS1", "HSV", 72, 255, lambda x: None)
        cv2.createTrackbar("LV1", "HSV", 42, 255, lambda x: None)
        cv2.createTrackbar("UH1", "HSV", 179, 179, lambda x: None)
        cv2.createTrackbar("US1", "HSV", 255, 255, lambda x: None)
        cv2.createTrackbar("UV1", "HSV", 255, 255, lambda x: None)
        cv2.createTrackbar("Iteration", "HSV", 0, 10, lambda x: None)
        cv2.createTrackbar("MinArea", "HSV", 11042, 50000, lambda x: None)
        cv2.createTrackbar("Kernel", "HSV", 5, 50, lambda x: None)

    def process(self, frame, pid_yaw):
        frame_h, frame_w = frame.shape[:2]
        center_x = frame_w // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # خواندن پارامترهای HSV از trackbar
        lh1 = cv2.getTrackbarPos("LH1", "HSV")
        ls1 = cv2.getTrackbarPos("LS1", "HSV")
        lv1 = cv2.getTrackbarPos("LV1", "HSV")
        uh1 = cv2.getTrackbarPos("UH1", "HSV")
        us1 = cv2.getTrackbarPos("US1", "HSV")
        uv1 = cv2.getTrackbarPos("UV1", "HSV")
        lower1 = np.array([lh1, ls1, lv1])
        upper1 = np.array([uh1, us1, uv1])

        mask1 = cv2.inRange(hsv, lower1, upper1)

        kernelTrack = cv2.getTrackbarPos("Kernel", "HSV")
        min_area = cv2.getTrackbarPos("MinArea", "HSV")
        iteration = cv2.getTrackbarPos("Iteration", "HSV")

        kernel = np.ones((max(1, kernelTrack), max(1, kernelTrack)), np.uint8)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=iteration)
        dialated = cv2.dilate(mask1, kernel, iterations=iteration)

        contours, _ = cv2.findContours(dialated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                    initial_state = np.array([[np.float32(cX)],
                                              [np.float32(cY)],
                                              [0],
                                              [0]], dtype=np.float32)
                    self.kf.kalman.statePre = initial_state.copy()
                    self.kf.kalman.statePost = initial_state.copy()
                else:
                    self.smoothed_center = (int(self.alpha * cX + (1 - self.alpha) * self.smoothed_center[0]),
                                            int(self.alpha * cY + (1 - self.alpha) * self.smoothed_center[1]))
                    self.kf.correct(self.smoothed_center[0], self.smoothed_center[1])

                cv2.drawContours(frame, [best_window], 0, (0, 255, 0), 3)
                cv2.circle(frame, self.smoothed_center, 5, (0, 0, 255), -1)
                cv2.line(frame, (center_x, frame_h // 2), self.smoothed_center, (255, 0, 0), 2)

                # محاسبه خطای افقی
                error_x = self.smoothed_center[0] - center_x
                error_threshold = 100

                # اگر فاصله زیاد بود yaw با حداقل سرعت حرکت کن
                min_yaw_speed = 10
                if abs(error_x) > error_threshold:
                    yaw_speed = np.sign(error_x) * min_yaw_speed
                else:
                    yaw_speed = pid_yaw.update(error_x)
                    yaw_speed = int(np.clip(yaw_speed, -100, 100))

                cv2.putText(frame, f"Yaw speed: {yaw_speed}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Error X: {error_x}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ارسال فرمان به تلو
                # tello.send_rc_control(0, 0, 0, yaw_speed)

                self.last_confidence = 1.0
                cv2.putText(frame, f"Center: {self.smoothed_center}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        else:
            self.last_confidence *= self.decay_factor
            cv2.putText(frame, f"Confidence: {self.last_confidence:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # tello.send_rc_control(0, 0, 0, 0)

        predicted = self.kf.predict()
        predicted = (int(predicted[0]), int(predicted[1]))
        cv2.circle(frame, predicted, 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Predicted: {predicted}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Mask1", mask1)

        return frame


class DroneController:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        # self.tello.takeoff()

        self.pid_yaw = PID(Kp=0.2, Ki=0.001, Kd=0.1)
        self.detector = WindowDetector()

    def run(self):
        while True:
            frame = self.tello.get_frame_read().frame
            frame = cv2.resize(frame, (640, 480))

            output = self.detector.process(frame, self.pid_yaw)
            cv2.imshow("Window Detection", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # self.tello.send_rc_control(0, 0, 0, 0)
                # self.tello.land()
                break

        cv2.destroyAllWindows()
        self.tello.streamoff()


if __name__ == "__main__":
    controller = DroneController()
    controller.run()
