import cv2
import numpy as np
import mediapipe as mp

class FaceRecognitionVerifier:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,  # Changed to static mode for better accuracy
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.threshold = 0.65  # Optimal balance between security and usability
        self.reference_features = None

    def extract_features(self, frame):
        """Enhanced feature extraction with 468 landmarks"""
        try:
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None
                
            # Extract all 468 3D landmarks
            landmarks = results.multi_face_landmarks[0].landmark
            return np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).flatten()
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None

    def verify(self, frame, reference_features):
        """Robust verification with normalization"""
        if reference_features is None:
            return False
            
        try:
            current_features = self.extract_features(frame)
            if current_features is None:
                return False

            # Normalization
            current_norm = current_features / np.linalg.norm(current_features)
            reference_norm = reference_features / np.linalg.norm(reference_features)
            
            # Cosine similarity with clipping
            similarity = np.dot(current_norm, reference_norm)
            similarity = np.clip(similarity, -1.0, 1.0)  # Prevent numerical errors
            
            # Debug output (uncomment to see values)
            # print(f"Similarity: {similarity:.2f}")
            
            return similarity > self.threshold
        except Exception as e:
            print(f"Verification error: {str(e)}")
            return False