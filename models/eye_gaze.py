import cv2
import numpy as np
import mediapipe as mp

class EyeGazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.EYE_ASPECT_RATIO_THRESHOLD = 0.25

    def detect(self, frame):
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                return True

            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            # Eye landmarks
            left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) 
                               for i in [362, 385, 387, 263, 373, 380]])
            right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) 
                                for i in [33, 160, 158, 133, 153, 144]])

            # Calculate eye aspect ratio
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            return bool(avg_ear > self.EYE_ASPECT_RATIO_THRESHOLD)
        except:
            return True

    def _eye_aspect_ratio(self, eye):
        # Vertical distances
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # Horizontal distance
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.25