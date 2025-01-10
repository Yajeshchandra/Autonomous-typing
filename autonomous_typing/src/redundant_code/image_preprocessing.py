#For testing purpose only



import cv2
import numpy as np
from sklearn.cluster import DBSCAN
image = cv2.imread('k.jpg')
        
resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

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

_, binarized = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

binarized = cv2.GaussianBlur(binarized, (5, 5), 0)
kernel = np.ones((5,5), np.uint8)   # 5x5 kernel
opening = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
binarized = cv2.dilate(binarized, kernel, iterations=3)

# Find coordinates of non-zero pixels
y_idxs, x_idxs = np.where(binarized == 255)
coords = np.column_stack((x_idxs, y_idxs))

# Perform DBSCAN clustering
db = DBSCAN(eps=100, min_samples=5).fit(coords)
labels = db.labels_

# Number of clusters, ignoring noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')

# Convert binarized image to BGR and dim it
binarized_color = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
binarized_color = cv2.multiply(binarized_color, np.array([0.5, 0.5, 0.5], dtype=np.float32))

# Optionally visualize clusters
for label in set(labels):
    if label == -1:
        color = (0, 0, 0)  # Black for noise
    else:
        color = tuple(np.random.randint(0, 255, 3).tolist())
    cluster_points = coords[labels == label]
    for x, y in cluster_points:
        cv2.circle(binarized_color, (x, y), 1, color, -1)

# Enclose the largest cluster in a bounding quadrilateral
if n_clusters > 0:
    largest_cluster_label = max(set(labels), key=lambda lbl: np.sum(labels == lbl) if lbl != -1 else 0)
    largest_cluster = coords[labels == largest_cluster_label]
    rect = cv2.minAreaRect(largest_cluster)
    box = cv2.boxPoints(rect)
    # box = np.int(box)
    box = np.array(box, dtype=np.int32).reshape((-1, 1, 2))

    cv2.polylines(binarized_color, [box], True, (0, 255, 0), 2)

# Apply dilation on colored clusters only
kernel = np.ones((5, 5), np.uint8)
# Create a mask for colored clusters (non-black)
mask = cv2.cvtColor(binarized_color, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
# Dilate the mask
dilated_mask = cv2.dilate(mask, kernel, iterations=100)
# Apply the dilated mask to the colored clusters
dilated_clusters = cv2.bitwise_and(binarized_color, binarized_color, mask=dilated_mask)

cv2.imwrite('dbscan_clusters_dilated.jpg', dilated_clusters)
cv2.imwrite('binarized.jpg', binarized)

