import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
import numpy as np
import threading
import time
import random

from models.face_count import FaceCountDetector
from models.face_recognition import FaceRecognitionVerifier
from models.head_pose import HeadPoseDetector
from models.eye_gaze import EyeGazeDetector
from models.tab_monitor import BrowserTabMonitor
from utils.snapshot import save_snapshot
from models.hotkey_monitor import HotkeyMonitor  # Added import

def snapshot_loop():
    while True:
        save_snapshot()
        time.sleep(120)

class InterviewMonitor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self.reference_face = None
        self.alert_cooldown = time.time()

        self.face_detector = FaceCountDetector()
        self.face_verifier = FaceRecognitionVerifier()
        self.head_pose_detector = HeadPoseDetector()
        self.eye_gaze_detector = EyeGazeDetector()
        self.tab_monitor = BrowserTabMonitor()
        self.hotkey_monitor = HotkeyMonitor()  # Added hotkey monitor
        
        # Start hotkey monitoring thread
        self.start_hotkey_monitoring()  # Added method call

        # Set expected tab ID
        self.tab_monitor.set_interview_tab_id("INTERVIEW_TAB_ID")

    # Added hotkey monitoring method
    def start_hotkey_monitoring(self):
        hotkey_thread = threading.Thread(
            target=self.hotkey_monitor.start_monitoring, 
            daemon=True
        )
        hotkey_thread.start()

    def capture_reference_face(self):
        print("Press 'S' to capture reference face")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            cv2.putText(frame, "Press 'S' to capture reference face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Reference Capture", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                face_count, _ = self.face_detector.detect(frame)
                if face_count == 1:
                    self.reference_face = self.face_verifier.extract_features(frame)
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("Error: Ensure exactly one face is visible")
            elif key == 27:
                self.running = False
        return False

    def nuclear_bool_conversion(self, result):
        if isinstance(result, np.ndarray):
            return bool(result.any() if result.size > 0 else False)
        return bool(result)

    def process_frame(self, frame):
        results = {
            'face_count': 0,
            'verified': False,
            'head_normal': True,
            'gaze_normal': True,
            'tab_suspicious': False
        }

        try:
            face_count, _ = self.face_detector.detect(frame)
            results['face_count'] = int(face_count)

            if self.reference_face is not None:
                verification_result = self.face_verifier.verify(frame, self.reference_face)
                results['verified'] = self.nuclear_bool_conversion(verification_result)

            results['head_normal'] = self.nuclear_bool_conversion(
                self.head_pose_detector.detect(frame)
            )
            results['gaze_normal'] = self.nuclear_bool_conversion(
                self.eye_gaze_detector.detect(frame)
            )

            # Simulate tab switch (test only)
            if random.random() < 0.02:
                self.tab_monitor.record_tab_change("OTHER_TAB_ID")
            else:
                self.tab_monitor.record_tab_change("INTERVIEW_TAB_ID")

            tab_stats = self.tab_monitor.get_tab_activity()
            results['tab_suspicious'] = tab_stats.get('is_suspicious', False)

        except Exception as e:
            print(f"[Frame Processing Error] {str(e)}")

        return results

    def run(self):
        try:
            if not self.capture_reference_face():
                return

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                results = self.process_frame(frame)

                status_text = [
                    f"Faces: {results['face_count']}",
                    f"Identity: {'Verified' if results['verified'] else 'Unknown'}",
                    f"Head: {'Normal' if results['head_normal'] else 'Abnormal'}",
                    f"Gaze: {'Normal' if results['gaze_normal'] else 'Abnormal'}",
                    f"Tabs: {'Suspicious' if results['tab_suspicious'] else 'Clean'}"
                ]

                for i, text in enumerate(status_text):
                    color = (0, 255, 0) if "Verified" in text or "Normal" in text or "Clean" in text else (0, 0, 255)
                    cv2.putText(frame, text, (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow("AI Proctor", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.hotkey_monitor.stop()  # Added cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            print("System shutdown complete")

if __name__ == "__main__":
    threading.Thread(target=snapshot_loop, daemon=True).start()
    monitor = InterviewMonitor()
    monitor.run()