import cv2
import numpy as np
import mediapipe as mp

class HeadPoseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # 3D model points (precise landmarks)
        self.model_points = np.array([
            [0.0, 0.0, 0.0],         # Nose tip
            [0.0, -330.0, -65.0],     # Chin
            [-225.0, 170.0, -135.0],  # Left eye
            [225.0, 170.0, -135.0],   # Right eye
            [-150.0, -150.0, -125.0], # Left mouth
            [150.0, -150.0, -125.0]   # Right mouth
        ], dtype=np.float32)
        
        self.thresholds = (30, 20, 20)  # yaw, pitch, roll

    def detect(self, frame):
        try:
            h, w = frame.shape[:2]
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                return True

            # Get precise landmarks
            landmarks = results.multi_face_landmarks[0].landmark
            indices = [1, 152, 263, 33, 287, 57]  # Nose, Chin, Eyes, Mouth
            image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices], dtype=np.float32)
            
            # Camera matrix
            camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
            
            # Solve PnP
            _, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # Get Euler angles
            rmat = cv2.Rodrigues(rvec)[0]
            angles = cv2.RQDecomp3x3(rmat)[0]
            yaw, pitch, roll = map(abs, angles)
            
            # Debug visualization (uncomment to see axes)
            # cv2.drawFrameAxes(frame, camera_matrix, None, rvec, tvec, 100)
            
            return yaw < self.thresholds[0] and pitch < self.thresholds[1] and roll < self.thresholds[2]
            
        except Exception as e:
            return True