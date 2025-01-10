import cv2
import numpy as np

# Constants for the actual dimensions of the image
TOTAL_LENGTH_MM = 354.076  # Length in mm
TOTAL_WIDTH_MM = 123.444   # Width in mm

def calculate_position(x, y, image_width, image_height):
    """Calculate the position of a point in mm based on pixel coordinates."""
    x_mm = (x / image_width) * TOTAL_LENGTH_MM
    y_mm = (y / image_height) * TOTAL_WIDTH_MM
    return x_mm, y_mm

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to capture and label points."""
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point and label
        points.append((x, y))
        label = f"P{len(points)}"
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Calculate position in mm
        x_mm, y_mm = x, y
        print(f"{label}: ({x_mm:.2f} mm, {y_mm:.2f} mm)")

# Load an image (user needs to provide an image)
image_path = "k.jpg"  # Replace with the path to your image
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load the image. Check the path.")
    exit()

# Dimensions of the image
image_height, image_width = image.shape[:2]

# Store points
points = []

# Create a window and set the mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback, param={'width': image_width, 'height': image_height})

print("Click on the image to select points. Press 'q' to quit.")

while True:
    # Show the image
    cv2.imshow("Image", image)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()

# Output all points
print("\nSelected Points:")
for i, (x, y) in enumerate(points):
    x_mm, y_mm = calculate_position(x, y, image_width, image_height)
    print(f"P{i + 1}: Pixel ({x}, {y}), Position ({x_mm:.2f} mm, {y_mm:.2f} mm)")
    

