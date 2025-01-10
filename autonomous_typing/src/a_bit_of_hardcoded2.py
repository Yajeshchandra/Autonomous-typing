#For testing purpose only


import numpy as np
import cv2
from ultralytics import YOLO


TOTAL_LENGTH_MM = 354.0  # Length in mm
TOTAL_WIDTH_MM = 123.4   # Width in mm

class Detector:
    def __init__(self, keyboard_width=TOTAL_LENGTH_MM, 
                     keyboard_height=TOTAL_WIDTH_MM, 
                     keys_dict=None,
                     model_path='yolov8x.pt', 
                     camera_matrix=None, 
                     dist_coeffs=None):
        
        self.keyboard_width = keyboard_width
        self.keyboard_height = keyboard_height
        self.keyboard_keys = keys_dict
        self.model = YOLO(model_path)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.key_positions_2d = None  # Dictionary of key: np.array([x, y])
        self.key_positions_3d = None  # Dictionary of key: np.array([x, y, z])
        self.transformation_matrix_2d = None
        self.transformation_matrix_3d = None
        
        
    def calibrate_camera(self,camera_matrix, dist_coeffs):
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1)) if dist_coeffs is None else np.array(dist_coeffs, dtype=np.float32)
        
    def load_key_positions(self, npy_path):
        self.key_positions_2d = np.load(npy_path, allow_pickle=True).item()
        
    def save_key_positions(self):
        if self.key_positions_2d is None:
            print('No key positions(2D) to save')
        else:
            np.save('key_positions_2d.npy', self.key_positions_2d)
        if self.key_positions_3d is None:
            print('No key positions(3D) to save')
        else:
            np.save('key_positions_3d.npy', self.key_positions_3d)    
        
    def calc_transform(self, corner_points_px):
        image_points = np.array(corner_points_px, dtype=np.float32)
        if image_points.shape != (4, 2):
            raise ValueError('Bounds for keyboard should be 4 corner points')
        # Define 3D model points for keyboard corners
        model_points = np.array([
            [0, 0, 0],  # Top-left
            [self.keyboard_width, 0, 0],  # Top-right
            [self.keyboard_width, self.keyboard_height, 0],  # Bottom-right
            [0, self.keyboard_height, 0]  # Bottom-left
        ], dtype=np.float32)
        print('Camera matrix is not defined')
        print('Only 2D key positions are available')
        transform_matrix = cv2.getPerspectiveTransform(
            image_points.astype(np.float32), 
            model_points.astype(np.float32)
        )
        self.transformation_matrix_2d = transform_matrix
        if self.camera_matrix is None:
            print('Camera matrix is not defined')
            print('Only 2D key positions are available')
            return
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
        self.transformation_matrix_3d = np.eye(4)
        self.transformation_matrix_3d[:3, :3] = rotation_matrix
        self.transformation_matrix_3d[:3, 3] = translation_vector.flatten()                
        
    def apply_transform(self):
        # Convert keys and 2D positions to arrays
        keys = np.array(list(self.keyboard_keys.keys()))
        key_pos = np.array(list(self.keyboard_keys.values()), dtype=np.float32)  # shape: (N, 2)
        
        # Create homogeneous coordinates for 2D transformation
        homogeneous_2d = np.column_stack([
            key_pos,                    # Original x,y coordinates
            np.ones(len(keys))         # Add homogeneous coordinate
        ])  # shape: (N, 3)
        
        # Apply 2D homogeneous transformation
        transformed_2d = np.dot(homogeneous_2d, self.transformation_matrix_2d.T)  # shape: (N, 3)
        
        # Convert back from homogeneous coordinates to 2D
        keys_pos_2d = transformed_2d[:, :2] / transformed_2d[:, 2:3]  # shape: (N, 2)
        
        # For 3D transformation, add z=0 to the 2D points
        homogeneous_3d = np.column_stack([
            key_pos,                    # Original x,y coordinates
            np.zeros(len(keys)),        # z = 0
            np.ones(len(keys))         # w = 1
        ])
        
        # Apply 3D transformation and extract x,y,z coordinates
        keys_pos_3d = np.dot(homogeneous_3d, self.transformation_matrix_3d.T)[:, :3]
        
        # Store results
        self.key_positions_2d = dict(zip(keys, keys_pos_2d))
        self.key_positions_3d = dict(zip(keys, keys_pos_3d))
        
    def detect_keyboard(self, ID=66, image_path=None):
    
        if image_path is None:
            print("Image path must be provided for image source.")
            return
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image.")
            return
        results = self.model(image)
        image_size = image.shape[:2]
        for det in results.boxes.data.tolist():
            cls = int(det[5])
            if cls == ID:
                x1, y1, x2, y2, conf, _ = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                roi = image[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                   pass 
                    
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Adjust coordinates relative to original image
                box[:, 0] += x1
                box[:, 1] += y1
                
                # Get axis-aligned bounding box
                x1 = np.min(box[:, 0])
                y1 = np.min(box[:, 1])
                x2 = np.max(box[:, 0])
                y2 = np.max(box[:, 1])
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"Class {cls}: {conf:.2f}"
                cv2.putText(image, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                        
                
        # Process results as needed
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def preprocessing(self):
        pass
    def tighter_bounds(self):
        pass
    def visualize(self):
        pass
    def dummy_bounds(self):
        corner_points = [
        [32, 21],   # Top-left
        [326, 21],   # Top-right
        [320, 83],   # Bottom-right
        [39, 83]    # Bottom-left
    ]
        return corner_points
    

