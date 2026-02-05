import cv2
import numpy as np

# Load image
img = cv2.imread('student_exams/6o/Giovanna Rafaela Francisco Amaral.jpg')

if img is None:
    print("Could not load image")
else:
    print(f"Image shape: {img.shape}")
    print(f"Height: {img.shape[0]}, Width: {img.shape[1]}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try simple thresholding first
    _, thresh_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Try Otsu thresholding
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply Hough Circle detection to find circles (bubbles)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=15,
        maxRadius=40
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"\nHough Circles detected: {len(circles[0])}")
        print("First 20 circles (x, y, radius):")
        for i, (x, y, r) in enumerate(circles[0][:20]):
            print(f"  {i+1}: ({x}, {y}), r={r}")
    else:
        print("\nNo circles detected with Hough transform")
    
    # Save threshold images for debugging
    cv2.imwrite('debug_thresh_simple.jpg', thresh_simple)
    cv2.imwrite('debug_thresh_otsu.jpg', thresh_otsu)
    
    # Draw circles on image copy
    if circles is not None:
        debug_circles = img.copy()
        for x, y, r in circles[0]:
            cv2.circle(debug_circles, (x, y), r, (0, 255, 0), 2)
        cv2.imwrite('debug_circles.jpg', debug_circles)
        print("\nSaved debug_circles.jpg, debug_thresh_simple.jpg, debug_thresh_otsu.jpg")
    
    # Also try finding ellipses using contours
    print("\n--- Trying to find filled regions ---")
    
    # Look at the answer grid area (approximately middle section)
    # Based on the contour analysis, the grid sections are around y=1800-3500
    grid_region = gray[1800:3600, 150:2550]
    
    # Apply threshold to grid region
    _, grid_thresh = cv2.threshold(grid_region, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in grid
    contours, _ = cv2.findContours(grid_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Contours in grid region: {len(contours)}")
    
    # Filter for circle-like shapes
    circle_like = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(c)
            if 0.5 < circularity < 1.5 and 100 < area < 3000 and 0.7 < w/h < 1.3:
                circle_like.append((x, y, w, h, area, circularity))
    
    print(f"Circle-like contours: {len(circle_like)}")
    
    # Draw on debug image
    grid_debug = cv2.cvtColor(grid_region, cv2.COLOR_GRAY2BGR)
    for x, y, w, h, area, circ in circle_like:
        cv2.rectangle(grid_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite('debug_grid_bubbles.jpg', grid_debug)
    print("Saved debug_grid_bubbles.jpg")
