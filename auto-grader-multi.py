import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
import os
from collections import defaultdict

# --- CONFIGURATION ---
# Uncomment and set path if on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

ROOT_DIR = 'STUDENT_EXAMS'

def imread_unicode(filepath):
    """
    Read image with Unicode filename support on Windows.
    OpenCV's imread doesn't handle Unicode paths well on Windows.
    """
    try:
        # First try normal imread
        img = cv2.imread(filepath)
        if img is not None:
            return img
    except:
        pass
    
    # Fallback: use numpy to read the file bytes
    try:
        with open(filepath, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"      Error reading {filepath}: {e}")
        return None

def imwrite_unicode(filepath, img):
    """
    Write image with Unicode filename support on Windows.
    OpenCV's imwrite doesn't handle Unicode paths well on Windows.
    """
    try:
        # Encode image to jpg format in memory
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, encoded = cv2.imencode('.jpg', img, encode_param)
        elif ext == '.png':
            _, encoded = cv2.imencode('.png', img)
        else:
            _, encoded = cv2.imencode('.jpg', img)
        
        # Write bytes to file with proper Unicode support
        with open(filepath, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    except Exception as e:
        print(f"      Error writing {filepath}: {e}")
        return False

def parse_answer_key_from_file(file_path):
    """
    Reads the answer key from a text file.
    Format: "1 - D" or "1. D" or "1 D" per line
    """
    if not os.path.exists(file_path):
        return None
    
    answer_key = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Match patterns like "1 - D", "1. D", "1 D", "1-D"
            match = re.match(r'(\d{1,2})\s*[\.\-]?\s*([A-Ea-e])', line)
            if match:
                q_num = int(match.group(1))
                ans = match.group(2).upper()
                if 1 <= q_num <= 25:
                    answer_key[q_num] = ans
    
    return answer_key if answer_key else None

def parse_answer_key(image_path):
    """
    Reads the Answer Key - first tries text file, then falls back to OCR on image.
    """
    # First, check for a text file version of the answer key
    class_folder = os.path.dirname(image_path)
    txt_file = os.path.join(class_folder, 'gabarito.txt')
    
    if os.path.exists(txt_file):
        print(f"   [Key] Found gabarito.txt - using text file")
        answer_key = parse_answer_key_from_file(txt_file)
        if answer_key:
            print(f"   [Key] Loaded {len(answer_key)} answers from text file")
            missing = [q for q in range(1, 26) if q not in answer_key]
            if missing:
                print(f"   [Key] WARNING: Missing answers for questions: {missing}")
            return answer_key
    
    print(f"   [Key] Reading gabarito image: {os.path.basename(image_path)}...")
    img = imread_unicode(image_path)
    if img is None: return {}
    
    # Preprocessing for OCR - use grayscale without harsh thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try OCR with page segmentation mode 6 (uniform block of text)
    # This works better for structured answer key layouts
    text = pytesseract.image_to_string(gray, config='--psm 6')
    
    # Debug: print OCR text
    print(f"   [Key] OCR detected text (first 500 chars):\n{text[:500]}")
    
    # More flexible regex: Finds "1. D", "1D", "1,A", "11A", "11.A", etc.
    # Allow for common OCR misreads like comma instead of period
    pattern = re.compile(r'(\d{1,2})\s*[\.\-,]?\s*([A-E])', re.IGNORECASE)
    
    answer_key = {}
    for match in pattern.finditer(text):
        q_num = int(match.group(1))
        ans = match.group(2).upper()
        # Only keep questions 1-25
        if 1 <= q_num <= 25:
            answer_key[q_num] = ans

    print(f"   [Key] Parsed {len(answer_key)} answers: {sorted(answer_key.keys())}")
    
    # Warn if missing questions
    missing = [q for q in range(1, 26) if q not in answer_key]
    if missing:
        print(f"   [Key] WARNING: Missing answers for questions: {missing}")
        print(f"   [Key] TIP: Create gabarito.txt with format '1 - D' per line for reliable parsing")
    
    # Manual override for "ANULO" keyword if found in text
    if "ANULO" in text.upper():
        print("   [Key] 'ANULO' detected. Check CSV for logic application.")
        
    return answer_key

def parse_question_tags(tags_file_path):
    """
    Reads question-tags.txt and returns a dictionary mapping question numbers to tags.
    Format: "1 - Tag1, Tag2"
    """
    if not os.path.exists(tags_file_path):
        print(f"   [Warning] No question-tags.txt found at {tags_file_path}")
        return {}
    
    question_tags = {}
    with open(tags_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '-' not in line:
                continue
            
            parts = line.split('-', 1)
            if len(parts) != 2:
                continue
            
            try:
                q_num = int(parts[0].strip())
                tags_str = parts[1].strip()
                
                # Parse comma-separated tags
                if tags_str:
                    tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                    if tags:
                        question_tags[q_num] = tags
            except ValueError:
                continue
    
    return question_tags

def cluster_positions(values, threshold):
    """Group values into clusters where consecutive values differ by < threshold."""
    if not values:
        return []
    sorted_vals = sorted(set(values))
    clusters = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] < threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    # Return cluster centers
    return [sum(c) / len(c) for c in clusters]


def find_answer_grid(circles, width, height, estimated_radius):
    """
    Find the answer bubble grid by looking for the densest region of circles.
    Returns (filtered_circles, grid_x_min, grid_x_max, grid_y_min, grid_y_max)
    """
    if len(circles) < 50:
        return circles, 0, width, 0, height
    
    # Use histogram to find dense regions in Y
    y_values = [c[1] for c in circles]
    y_min, y_max = min(y_values), max(y_values)
    
    # Find the region with highest density of circles
    # The answer grid should be a dense rectangular region
    bin_size = estimated_radius * 3
    
    # Create density map
    y_bins = int((y_max - y_min) / bin_size) + 1
    y_counts = [0] * y_bins
    
    for y in y_values:
        bin_idx = int((y - y_min) / bin_size)
        if 0 <= bin_idx < y_bins:
            y_counts[bin_idx] += 1
    
    # Find contiguous region with high density
    threshold = max(y_counts) * 0.3  # At least 30% of peak density
    
    # Find first and last bins above threshold
    first_bin = 0
    for i, count in enumerate(y_counts):
        if count >= threshold:
            first_bin = i
            break
    
    last_bin = y_bins - 1
    for i in range(y_bins - 1, -1, -1):
        if y_counts[i] >= threshold:
            last_bin = i
            break
    
    grid_y_min = y_min + first_bin * bin_size - bin_size
    grid_y_max = y_min + (last_bin + 1) * bin_size + bin_size
    
    # Filter circles within this Y range
    filtered = [c for c in circles if grid_y_min <= c[1] <= grid_y_max]
    
    return filtered, 0, width, grid_y_min, grid_y_max


def parse_student_sheet(image_path, save_debug=True):
    """
    Uses Hough Circle detection + Otsu thresholding for cropped answer sheets.
    Layout: Vertical list of 30 questions, each with 5 bubbles (A-E) horizontally.
    We only grade questions 1-25.
    Returns (student_answers, debug_image)
    """
    img = imread_unicode(image_path)
    if img is None: 
        return {}, None

    debug_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    print(f"      Image: {width}x{height}")
    
    # Apply Otsu thresholding to get binary image
    # This makes filled bubbles clearly distinguishable from unfilled ones
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # For cropped images (~400px wide), use appropriate parameters
    min_radius = 10
    max_radius = 25
    min_dist = 20
    
    # Detect circles using Hough transform on the grayscale image
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=50,
        param2=25,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is None:
        print(f"      WARNING: No circles detected")
        return {}, debug_img
    
    circles = np.uint16(np.around(circles))
    all_circles = [(int(x), int(y), int(r)) for x, y, r in circles[0]]
    print(f"      Detected {len(all_circles)} circles")
    
    # Draw all circles in light blue
    for x, y, r in all_circles:
        cv2.circle(debug_img, (x, y), r, (255, 200, 100), 1)
    
    # Step 1: Find the 5 option columns (A-E) by clustering X positions
    # Filter to main bubble area (X > 15% of width to exclude noise on left edge)
    main_circles = [c for c in all_circles if c[0] > width * 0.15]
    
    if len(main_circles) < 25:
        print(f"      WARNING: Too few circles in main area ({len(main_circles)})")
        return {}, debug_img
    
    # Find X column centers using histogram-based approach
    # Create histogram of X positions
    x_vals = [c[0] for c in main_circles]
    x_min, x_max = min(x_vals), max(x_vals)
    
    # The 5 columns should span most of the image width
    # Each column should have ~30 circles (one per question row)
    # Use a histogram with small bins to find peaks
    bin_width = 10
    n_bins = int((x_max - x_min) / bin_width) + 1
    histogram = [0] * n_bins
    
    for x in x_vals:
        bin_idx = int((x - x_min) / bin_width)
        if 0 <= bin_idx < n_bins:
            histogram[bin_idx] += 1
    
    # Find peaks (local maxima with count > 5)
    peaks = []
    for i in range(1, n_bins - 1):
        if histogram[i] >= 5 and histogram[i] >= histogram[i-1] and histogram[i] >= histogram[i+1]:
            peak_x = x_min + i * bin_width + bin_width / 2
            peaks.append((peak_x, histogram[i]))
    
    # Also check first and last bins
    if histogram[0] >= 5:
        peaks.append((x_min + bin_width / 2, histogram[0]))
    if histogram[n_bins-1] >= 5:
        peaks.append((x_max - bin_width / 2, histogram[n_bins-1]))
    
    # Sort peaks by X position
    peaks = sorted(peaks, key=lambda p: p[0])
    
    # Merge nearby peaks (within 2 bin widths)
    merged_peaks = []
    for peak in peaks:
        if not merged_peaks or peak[0] - merged_peaks[-1][0] > bin_width * 2:
            merged_peaks.append(peak)
        else:
            # Merge with previous: weighted average by count
            prev_x, prev_count = merged_peaks[-1]
            new_x = (prev_x * prev_count + peak[0] * peak[1]) / (prev_count + peak[1])
            merged_peaks[-1] = (new_x, prev_count + peak[1])
    
    # Take the 5 peaks with highest counts
    if len(merged_peaks) > 5:
        # Sort by count, take top 5, then sort by X
        merged_peaks = sorted(merged_peaks, key=lambda p: -p[1])[:5]
        merged_peaks = sorted(merged_peaks, key=lambda p: p[0])
    
    x_centers = [p[0] for p in merged_peaks]
    
    # If we don't have exactly 5 columns, try to interpolate/extrapolate
    if len(x_centers) >= 2 and len(x_centers) < 5:
        # Estimate column spacing from detected columns
        spacings = [x_centers[i+1] - x_centers[i] for i in range(len(x_centers)-1)]
        avg_spacing = sum(spacings) / len(spacings)
        
        # Extrapolate to get 5 columns
        while len(x_centers) < 5:
            # Try adding to left or right based on where there's more space
            left_space = x_centers[0] - x_min
            right_space = x_max - x_centers[-1]
            
            if left_space > avg_spacing * 0.7:
                x_centers.insert(0, x_centers[0] - avg_spacing)
            elif right_space > avg_spacing * 0.7:
                x_centers.append(x_centers[-1] + avg_spacing)
            else:
                break
    
    print(f"      Found {len(x_centers)} option columns at X: {[int(x) for x in x_centers]}")
    
    if len(x_centers) < 4:
        print(f"      WARNING: Not enough columns detected")
        return {}, debug_img
    
    # Step 2: Assign each circle to its nearest X column
    def get_column(x):
        if not x_centers:
            return -1
        min_dist_val = float('inf')
        col = -1
        for i, center in enumerate(x_centers):
            d = abs(x - center)
            if d < min_dist_val and d < 50:  # Within 50px of column center (more flexible)
                min_dist_val = d
                col = i
        return col
    
    # Step 3: Group circles by Y position into rows (questions)
    circles_sorted_y = sorted(main_circles, key=lambda c: c[1])
    
    rows = []
    current_row = [circles_sorted_y[0]]
    
    for i in range(1, len(circles_sorted_y)):
        curr = circles_sorted_y[i]
        prev_avg_y = sum(c[1] for c in current_row) / len(current_row)
        # Same row if Y is close (within 25px of row average)
        if abs(curr[1] - prev_avg_y) < 25:
            current_row.append(curr)
        else:
            if len(current_row) >= 3:  # Valid row has at least 3 circles
                rows.append(current_row)
            current_row = [curr]
    if len(current_row) >= 3:
        rows.append(current_row)
    
    # Sort rows by Y position (top to bottom = Q1 to Q30)
    rows = sorted(rows, key=lambda r: sum(c[1] for c in r) / len(r))
    
    print(f"      Found {len(rows)} question rows")
    
    # Calculate average radius from all detected circles
    avg_radius = int(sum(c[2] for c in main_circles) / len(main_circles)) if main_circles else 15
    
    # Step 4: Process each row as a question (only first 25)
    student_answers = {}
    
    for q_idx, row in enumerate(rows[:25]):
        question_num = q_idx + 1
        
        # Calculate row's Y center
        row_y = int(sum(c[1] for c in row) / len(row))
        
        # Assign each detected circle to a column (A=0, B=1, C=2, D=3, E=4)
        detected_options = {}  # column_index -> circle
        for circle in row:
            col = get_column(circle[0])
            if col >= 0:
                if col not in detected_options:
                    detected_options[col] = circle
        
        # Build complete options dict - use detected circles OR interpolated positions
        # This ensures we check all 5 columns even if Hough missed some circles
        options = {}
        for col_idx in range(len(x_centers)):
            if col_idx in detected_options:
                options[col_idx] = detected_options[col_idx]
            else:
                # Interpolate: use column center X and row's Y
                cx = int(x_centers[col_idx])
                cy = row_y
                cr = avg_radius
                options[col_idx] = (cx, cy, cr)
        
        if len(options) < 3:
            # Not enough columns detected
            continue
        
        # Find which bubble is filled using Otsu thresholded image
        # In the thresholded image: 255 = filled (black in original), 0 = unfilled (white in original)
        max_fill = 0
        second_fill = 0
        detected_letter = ""
        detected_circle = None
        
        for col_idx, (cx, cy, cr) in options.items():
            # Create mask for this circle
            mask = np.zeros(thresh.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), cr, 255, -1)
            
            # Calculate mean of thresholded image within circle
            # Higher value = more filled (white pixels in thresh = filled bubble)
            fill_ratio = cv2.mean(thresh, mask=mask)[0]
            
            if fill_ratio > max_fill:
                second_fill = max_fill
                max_fill = fill_ratio
                detected_letter = chr(ord('A') + col_idx)
                detected_circle = (cx, cy, cr)
            elif fill_ratio > second_fill:
                second_fill = fill_ratio
        
        # With Otsu thresholding, filled bubbles should have high fill_ratio
        # and be significantly different from unfilled bubbles
        fill_diff = max_fill - second_fill
        
        # Simplified logic: just check that the max fill is high enough 
        # and there's a reasonable difference from the second highest
        # A filled bubble typically has fill_ratio > 80 and stands out from others
        is_filled = (max_fill > 100 and fill_diff > 5) or (max_fill > 60 and fill_diff > 15)
        
        if is_filled and detected_circle:
            student_answers[question_num] = detected_letter
            
            # Draw detected answer in green
            cx, cy, cr = detected_circle
            cv2.circle(debug_img, (cx, cy), cr, (0, 255, 0), 3)
            cv2.putText(debug_img, f"Q{question_num}:{detected_letter}", 
                       (cx + cr + 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # Mark as uncertain
            if row:
                cx, cy, cr = row[0]
                cv2.putText(debug_img, f"Q{question_num}:?", 
                           (cx + cr + 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    print(f"      Detected answers for {len(student_answers)} questions")
    
    # Create Otsu debug image with circles overlaid
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for x, y, r in all_circles:
        cv2.circle(thresh_color, (x, y), r, (255, 200, 100), 1)
    for q_num, letter in student_answers.items():
        # Find the circle for this answer
        for row in rows[:25]:
            row_y = sum(c[1] for c in row) / len(row)
            q_idx = rows.index(row)
            if q_idx + 1 == q_num:
                for circle in row:
                    col = get_column(circle[0])
                    if col >= 0 and chr(ord('A') + col) == letter:
                        cx, cy, cr = circle
                        cv2.circle(thresh_color, (cx, cy), cr, (0, 255, 0), 2)
                break
    
    return student_answers, debug_img, thresh_color


def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Folder '{ROOT_DIR}' not found.")
        return

    # Iterate through class folders (6o, 7oA, 7oV...)
    for class_folder in os.listdir(ROOT_DIR):
        class_path = os.path.join(ROOT_DIR, class_folder)
        
        if not os.path.isdir(class_path):
            continue

        print(f"\nProcessing Class: {class_folder}")
        print("=" * 30)

        # 1. Identify Gabarito vs Students
        gabarito_file = None
        student_files = []
        
        for f in os.listdir(class_path):
            if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            # Check for 'gabarito' in filename (case insensitive)
            if 'gabarito' in f.lower():
                gabarito_file = os.path.join(class_path, f)
            else:
                student_files.append(os.path.join(class_path, f))

        if not gabarito_file:
            print(f"Skipping {class_folder}: No file named 'gabarito' found.")
            continue

        # 2. Parse Answer Key
        class_key = parse_answer_key(gabarito_file)
        if not class_key:
            print(f"Skipping {class_folder}: Could not read answer key.")
            continue

        # 3. Parse Question Tags
        tags_file = os.path.join(class_path, 'question-tags.txt')
        question_tags = parse_question_tags(tags_file)
        print(f"   [Tags] Loaded {len(question_tags)} questions with tags")

        # 4. Initialize tracking structures
        class_results = []
        
        # 5. Create debug output folder
        debug_folder = os.path.join(class_path, 'debug_output')
        os.makedirs(debug_folder, exist_ok=True)
        
        # 6. Process Students for this Class
        for s_file in student_files:
            s_name = os.path.splitext(os.path.basename(s_file))[0]
            print(f"   -> Grading: {s_name}")
            
            result = parse_student_sheet(s_file)
            answers = result[0]
            debug_img = result[1] if len(result) > 1 else None
            thresh_img = result[2] if len(result) > 2 else None
            
            # Save debug image
            if debug_img is not None:
                debug_path = os.path.join(debug_folder, f"{s_name}_debug.jpg")
                imwrite_unicode(debug_path, debug_img)
                print(f"      Saved debug image: {os.path.basename(debug_path)}")
            
            # Save Otsu thresholded image
            if thresh_img is not None:
                thresh_path = os.path.join(debug_folder, f"{s_name}_otsu.jpg")
                imwrite_unicode(thresh_path, thresh_img)
                print(f"      Saved Otsu image: {os.path.basename(thresh_path)}")
            
            score = 0
            missed_tags = set()
            # Dictionary starting with Name
            student_data = {'Name': s_name}
            
            # Compare every question in the Key (limit to questions 1-25)
            for q_num, correct_ans in class_key.items():
                # Skip questions beyond 25
                if q_num > 25:
                    continue
                    
                student_ans = answers.get(q_num, "")
                
                # Check for ANULO/Annulled (*) or correct match
                is_correct = (correct_ans == '*' or student_ans == correct_ans)
                
                if is_correct:
                    score += 1
                else:
                    # Track missed tags
                    if q_num in question_tags:
                        for tag in question_tags[q_num]:
                            missed_tags.add(tag)
                    
                # Save answer detail
                student_data[f'Q{q_num}'] = student_ans

            student_data['Total Score'] = score
            student_data['Missed Tags'] = ', '.join(sorted(missed_tags)) if missed_tags else 'None'
            class_results.append(student_data)

        # 7. Save Class CSV (sorted alphabetically by name)
        if class_results:
            # Sort results alphabetically by student name
            class_results.sort(key=lambda x: x['Name'].lower())
            
            df = pd.DataFrame(class_results)
            
            # Reorder columns: Name, Score, Missed Tags, Q1, Q2...
            cols = ['Name', 'Total Score', 'Missed Tags'] + [c for c in df.columns if c not in ['Name', 'Total Score', 'Missed Tags']]
            df = df[cols]
            
            output_filename = f"results_{class_folder}.csv"
            df.to_csv(output_filename, index=False)
            print(f"   [Done] Saved {output_filename}")
        else:
            print(f"   [Info] No students found in {class_folder}")

    print("\n✅ All classes processed. CSV files generated.")

if __name__ == "__main__":
    main()
