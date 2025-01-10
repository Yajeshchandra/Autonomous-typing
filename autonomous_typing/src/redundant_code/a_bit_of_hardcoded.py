#For testing purpose only

import numpy as np
import cv2

TOTAL_LENGTH_MM = 354.076  # Length in mm
TOTAL_WIDTH_MM = 123.444   # Width in mm

class KeyboardPositionCalculator:
    def __init__(self, keyboard_width_mm=TOTAL_LENGTH_MM, keyboard_height_mm=TOTAL_WIDTH_MM):
        """
        Initialize the keyboard calculator with physical dimensions.
        
        Args:
            keyboard_width_mm: Physical width of keyboard in mm
            keyboard_height_mm: Physical height of keyboard in mm
        """
        self.keyboard_width = keyboard_width_mm
        self.keyboard_height = keyboard_height_mm
        self.camera_matrix = None
        self.dist_coeffs = None
        self.key_positions_2d = None  # Dictionary of key: np.array([x, y])
        self.key_positions_3d = None  # Dictionary of key: np.array([x, y, z])
        self.transformation_matrix = None
        
    def load_camera_calibration(self, camera_matrix, dist_coeffs=None):
        """Load camera calibration parameters."""
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1)) if dist_coeffs is None else np.array(dist_coeffs, dtype=np.float32)
        
    def load_key_positions(self, npy_path):
        """
        Load key positions from NPY file.
        Expects dictionary of key: np.array([x, y])
        """
        self.key_positions_2d = np.load(npy_path, allow_pickle=True).item()
            
    def calculate_transformation_matrix(self, corner_points_px):
        """
        Calculate transformation matrix from keyboard corners in image.
        
        Args:
            corner_points_px: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                            in pixel coordinates, ordered: top-left, top-right, bottom-right, bottom-left
        """
        if self.camera_matrix is None:
            raise ValueError("Camera calibration must be loaded first")
            
        # Convert corner points to numpy array
        image_points = np.array(corner_points_px, dtype=np.float32)
        
        # Define 3D model points for keyboard corners
        model_points = np.array([
            [0, 0, 0],  # Top-left
            [self.keyboard_width, 0, 0],  # Top-right
            [self.keyboard_width, self.keyboard_height, 0],  # Bottom-right
            [0, self.keyboard_height, 0]  # Bottom-left
        ], dtype=np.float32)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            raise RuntimeError("Failed to solve PnP")
            
        # Convert rotation vector to matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Create transformation matrix
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[:3, :3] = rotation_matrix
        self.transformation_matrix[:3, 3] = translation_vector.flatten()
        
    def calculate_all_key_positions(self):
        """Calculate 3D positions for all keys."""
        if self.transformation_matrix is None:
            raise ValueError("Transformation matrix must be calculated first")
        
        # Initialize 3D positions dictionary
        self.key_positions_3d = {}
        
        # Stack all 2D positions for vectorized transformation
        keys = list(self.key_positions_2d.keys())
        positions_2d = np.stack([self.key_positions_2d[k] for k in keys])
        
        # Create homogeneous coordinates (add z=0 and w=1)
        homogeneous_points = np.column_stack([
            positions_2d,
            np.zeros(len(keys)),
            np.ones(len(keys))
        ])
        
        # Transform all points at once
        positions_3d = np.dot(homogeneous_points, self.transformation_matrix.T)
        
        # Store results in dictionary
        for idx, key in enumerate(keys):
            self.key_positions_3d[key] = positions_3d[idx, :3]  # Only store x,y,z
        
    def save_positions(self, filepath):
        """Save calculated 3D positions to binary file."""
        if self.key_positions_3d is None:
            raise ValueError("3D positions must be calculated first")
        np.save(filepath, self.key_positions_3d)
        
    def load_positions(self, filepath):
        """Load 3D positions from binary file."""
        self.key_positions_3d = np.load(filepath, allow_pickle=True).item()
        
    def get_key_position(self, key):
        """Get 3D position of a specific key."""
        if self.key_positions_3d is None:
            raise ValueError("3D positions must be calculated first")
        if key not in self.key_positions_3d:
            raise KeyError(f"Key '{key}' not found")
        return self.key_positions_3d[key]
    

def main():
    # Example usage
    calculator = KeyboardPositionCalculator()
    
    # Load camera calibration (example values)
    camera_matrix = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])
    calculator.load_camera_calibration(camera_matrix)
    
    # Load key positions from NPY file
    calculator.load_key_positions('keyboard_layout.npy')
    
    # Example corner points in image (would come from camera)
    corner_points = [
        [32, 21],   # Top-left
        [326, 21],   # Top-right
        [320, 83],   # Bottom-right
        [39, 83]    # Bottom-left
    ]
    
    # Calculate transformation and 3D positions
    calculator.calculate_transformation_matrix(corner_points)
    calculator.calculate_all_key_positions()
    
    # Save positions
    calculator.save_positions('keyboard_3d_positions.npy')
    
    # Get position of a specific key
    print("Position of 'A' key:", calculator.get_key_position('A'))

if __name__ == "__main__":
    main()
    
    
from ultralytics import YOLO

class KeyboardDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize keyboard detector with YOLO model.
        
        Args:
            model_path: Path to YOLO model weights (defaults to YOLOv8 nano)
        """
        self.model = YOLO(model_path)
        # Keyboard class ID in COCO dataset is 76
        self.keyboard_class_id = 76
        
    def detect_keyboard(self, frame):
        """
        Detect keyboard in frame and return bounding box coordinates.
        
        Args:
            frame: Input image/frame (BGR format)
            
        Returns:
            bbox: List of [x1, y1, x2, y2] or None if no keyboard detected
            confidence: Detection confidence or None if no keyboard detected
        """
        # Run inference
        results = self.model(frame, verbose=False)[0]
        
        # Find keyboard detections
        keyboard_detections = []
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == self.keyboard_class_id:
                keyboard_detections.append((conf, [x1, y1, x2, y2]))
        
        # Return highest confidence detection if any
        if keyboard_detections:
            keyboard_detections.sort(reverse=True)  # Sort by confidence
            return keyboard_detections[0][1], keyboard_detections[0][0]
        
        return None, None
    
    def draw_detection(self, frame, bbox, confidence):
        """
        Draw bounding box and confidence score on frame.
        
        Args:
            frame: Input image/frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            confidence: Detection confidence
            
        Returns:
            frame: Frame with drawn detection
        """
        if bbox is None:
            return frame
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence score
        text = f"Keyboard: {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def get_corners(self, bbox):
        """
        Get corner points from bounding box in clock-wise order:
        top-left, top-right, bottom-right, bottom-left
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            corners: List of corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        if bbox is None:
            return None
            
        x1, y1, x2, y2 = map(int, bbox)
        return [
            [x1, y1],  # Top-left
            [x2, y1],  # Top-right
            [x2, y2],  # Bottom-right
            [x1, y2]   # Bottom-left
        ]

# def main():
#     # Initialize detector
#     detector = KeyboardDetector()
    
#     # Open camera
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Detect keyboard
#         bbox, conf = detector.detect_keyboard(frame)
        
#         # Draw detection
#         frame = detector.draw_detection(frame, bbox, conf)
        
#         # Get corner points if needed
#         if bbox is not None:
#             corners = detector.get_corners(bbox)
#             # Draw corners
#             for corner in corners:
#                 cv2.circle(frame, tuple(corner), 5, (0, 0, 255), -1)
        
#         # Show frame
#         cv2.imshow('Keyboard Detection', frame)
        
#         # Exit on 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

