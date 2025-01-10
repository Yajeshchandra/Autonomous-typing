#For testing purpose only



import cv2
import numpy as np

def load_and_preprocess(image_path,mode=0):
    """Load and preprocess image for template matching."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Convert to grayscale
    resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    
    if mode==0 : 
        cv2.imwrite(f"{image_path}_gray.jpg", gray)
        return image, gray
    

    # Apply a large Gaussian blur to estimate the background
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    # Subtract the background
    background_subtracted = cv2.subtract(gray, blur)
    
    normalized = cv2.normalize(background_subtracted, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(normalized)

    # Estimate the illumination
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    illumination = cv2.morphologyEx(clahe_result, cv2.MORPH_CLOSE, kernel)
    # Correct the image by dividing by the illumination
    illumination_corrected = cv2.divide(clahe_result, illumination, scale=255)

    _, binarized = cv2.threshold(illumination_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{image_path}_binarized.jpg", binarized)
    return image, binarized
        

def detect_keyboard(image_path, template_path, threshold=0.8):
    """
    Detect keyboard in image using template matching.
    
    Args:
        image_path: Path to the image to search in
        template_path: Path to the template keyboard image
        threshold: Matching threshold (0-1), higher means more strict matching
    
    Returns:
        List of (x, y, w, h) coordinates for matches
    """
    # Load and preprocess both images
    image, image_processed = load_and_preprocess(image_path)
    template, template_processed = load_and_preprocess(template_path)
    
    # Get template dimensions
    h, w = template_processed.shape
    
    # Perform template matching
    result = cv2.matchTemplate(image_processed, template_processed, cv2.TM_CCOEFF_NORMED)
    
    # Find locations above threshold
    locations = np.where(result >= threshold)
    matches = []
    
    # Convert locations to x,y coordinates
    for pt in zip(*locations[::-1]):
        matches.append((pt[0], pt[1], w, h))
    
    return image, matches

def draw_detections(image, matches):
    """Draw rectangles around detected keyboards."""
    result = image.copy()
    for (x, y, w, h) in matches:
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return result

def main():
    # Example usage
    image_path = "26.jpg"
    template_path = "key.png"
    
    try:
        # Detect keyboards
        image, matches = detect_keyboard(image_path, template_path, threshold=0.4)
        
        if matches:
            # Draw detections
            result = draw_detections(image, matches)
            
            # Display results
            cv2.imshow("Keyboard Detections", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print(f"Found {len(matches)} potential keyboard(s)")
        else:
            print("No keyboards detected")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()