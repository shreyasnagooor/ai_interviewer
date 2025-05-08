import cv2
import numpy as np
import mediapipe as mp

class FaceRecognitionVerifier:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,  # Use static mode for accuracy
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.threshold = 0.80  # Increased threshold for stricter verification

    def extract_features(self, frame):
        """Extracts 3D landmarks as feature vector"""
        try:
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                print("No face landmarks found during extraction.")
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            return np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).flatten()
        except Exception as e:
            print(f"[Error] Feature extraction failed: {str(e)}")
            return None

    def verify(self, frame, reference_features):
        """Compares current frame with reference features using cosine similarity"""
        if reference_features is None:
            print("[Warning] Reference features not set.")
            return False

        try:
            current_features = self.extract_features(frame)
            if current_features is None:
                print("[Warning] Could not extract current features.")
                return False

            current_norm = current_features / np.linalg.norm(current_features)
            reference_norm = reference_features / np.linalg.norm(reference_features)

            similarity = np.dot(current_norm, reference_norm)
            similarity = np.clip(similarity, -1.0, 1.0)  # Prevent math errors

            print(f"[Debug] Cosine similarity: {similarity:.4f}")

            return similarity > self.threshold
        except Exception as e:
            print(f"[Error] Verification failed: {str(e)}")
            return False
