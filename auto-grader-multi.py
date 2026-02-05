import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
import os
from collections import defaultdict
from datetime import datetime

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

def generate_dashboard(all_class_data, output_file='dashboard.html'):
    """
    Generates an enhanced HTML dashboard with:
    - Filter bar for class, score range, and search
    - Interactive charts using Chart.js
    - Student statistics with debug and Otsu images
    - Question and tag accuracy tables
    - Modern responsive UI
    """
    
    # Prepare data for JavaScript
    import json
    
    # Build JSON data for charts and filters
    chart_data = {}
    all_students_json = []
    
    for class_name, class_info in all_class_data.items():
        students = class_info['students']
        question_stats = class_info['question_stats']
        tag_stats = class_info['tag_stats']
        total_questions = class_info['total_questions']
        
        # Score distribution for histogram
        scores = [s['score'] for s in students]
        chart_data[class_name] = {
            'scores': scores,
            'avg': sum(scores) / len(scores) if scores else 0,
            'question_accuracy': {str(q): (s['correct'] / (s['correct'] + s['incorrect']) * 100) if (s['correct'] + s['incorrect']) > 0 else 0 
                                  for q, s in question_stats.items()},
            'tag_accuracy': {t: (s['correct'] / (s['correct'] + s['incorrect']) * 100) if (s['correct'] + s['incorrect']) > 0 else 0 
                            for t, s in tag_stats.items()}
        }
        
        for student in students:
            all_students_json.append({
                'name': student['name'],
                'class': class_name,
                'score': student['score'],
                'total': total_questions,
                'percentage': round(student['score'] / total_questions * 100, 1) if total_questions > 0 else 0,
                'missed_tags': list(student['missed_tags']),
                'answers': student.get('answers', {})
            })
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Exam Results Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #4f46e5;
            --primary-dark: #3730a3;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-600: #4b5563;
            --gray-800: #1f2937;
            --gray-900: #111827;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            color: white;
            padding: 30px 0;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        header .subtitle {{
            opacity: 0.9;
            font-size: 1.1rem;
        }}
        
        /* Filter Bar */
        .filter-bar {{
            background: white;
            border-radius: 16px;
            padding: 20px 30px;
            margin-bottom: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
        }}
        
        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .filter-group label {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--gray-600);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .filter-group select,
        .filter-group input {{
            padding: 10px 15px;
            border: 2px solid var(--gray-200);
            border-radius: 8px;
            font-size: 0.95rem;
            min-width: 180px;
            transition: all 0.2s;
        }}
        
        .filter-group select:focus,
        .filter-group input:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }}
        
        .filter-stats {{
            margin-left: auto;
            display: flex;
            gap: 30px;
        }}
        
        .filter-stat {{
            text-align: center;
        }}
        
        .filter-stat-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
        }}
        
        .filter-stat-label {{
            font-size: 0.75rem;
            color: var(--gray-600);
            text-transform: uppercase;
        }}
        
        /* Tabs */
        .tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 12px;
            padding: 5px;
        }}
        
        .tab {{
            padding: 12px 24px;
            border: none;
            background: transparent;
            color: white;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.2s;
        }}
        
        .tab:hover {{
            background: rgba(255,255,255,0.2);
        }}
        
        .tab.active {{
            background: white;
            color: var(--primary);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        /* Content Sections */
        .section {{
            display: none;
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            margin-bottom: 25px;
        }}
        
        .section.active {{
            display: block;
        }}
        
        .section-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--gray-800);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        /* Charts Grid */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        
        .chart-card {{
            background: var(--gray-50);
            border-radius: 12px;
            padding: 20px;
        }}
        
        .chart-card h3 {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--gray-800);
            margin-bottom: 15px;
        }}
        
        /* Stats Cards */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
        }}
        
        .stat-card.success {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }}
        
        .stat-card.warning {{
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        }}
        
        .stat-card.danger {{
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        
        /* Student Cards */
        .students-grid {{
            display: grid;
            gap: 20px;
        }}
        
        .student-card {{
            background: var(--gray-50);
            border-radius: 12px;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 20px;
            align-items: center;
            transition: all 0.2s;
            border: 2px solid transparent;
            cursor: pointer;
        }}
        
        .student-card:hover {{
            border-color: var(--primary);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
        }}
        
        .student-card.expanded {{
            grid-template-columns: 1fr;
            border-color: var(--primary);
        }}
        
        .student-card.expanded .student-main {{
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 20px;
            align-items: center;
        }}
        
        .student-main {{
            display: contents;
        }}
        
        .student-card.expanded .student-main {{
            display: grid;
        }}
        
        .student-details {{
            display: none;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 2px solid var(--gray-200);
        }}
        
        .student-card.expanded .student-details {{
            display: block;
        }}
        
        .missed-tags-full {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}
        
        .missed-tag {{
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        .no-missed {{
            color: var(--success);
            font-weight: 600;
        }}
        
        .student-info h4 {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--gray-800);
            margin-bottom: 5px;
        }}
        
        .student-info .class-badge {{
            display: inline-block;
            background: var(--primary);
            color: white;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 10px;
        }}
        
        .student-info .tags {{
            color: var(--gray-600);
            font-size: 0.85rem;
            margin-top: 8px;
        }}
        
        .student-score {{
            text-align: center;
            min-width: 100px;
        }}
        
        .score-circle {{
            width: 70px;
            height: 70px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            font-weight: 700;
            color: white;
            margin: 0 auto 5px;
        }}
        
        .score-circle.high {{ background: var(--success); }}
        .score-circle.medium {{ background: var(--warning); }}
        .score-circle.low {{ background: var(--danger); }}
        
        .score-label {{
            font-size: 0.8rem;
            color: var(--gray-600);
        }}
        
        .student-images {{
            display: flex;
            gap: 10px;
        }}
        
        .student-images img {{
            width: 120px;
            height: 180px;
            object-fit: cover;
            border-radius: 8px;
            border: 2px solid var(--gray-200);
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .student-images img:hover {{
            transform: scale(1.05);
            border-color: var(--primary);
        }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: var(--gray-800);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid var(--gray-200);
        }}
        
        tr:hover {{
            background: var(--gray-50);
        }}
        
        .accuracy-bar {{
            background: var(--gray-200);
            border-radius: 100px;
            height: 8px;
            overflow: hidden;
            min-width: 150px;
        }}
        
        .accuracy-fill {{
            height: 100%;
            border-radius: 100px;
            transition: width 0.5s ease;
        }}
        
        .accuracy-fill.high {{ background: var(--success); }}
        .accuracy-fill.medium {{ background: var(--warning); }}
        .accuracy-fill.low {{ background: var(--danger); }}
        
        /* Modal */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            backdrop-filter: blur(5px);
        }}
        
        .modal-content {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
            border-radius: 8px;
        }}
        
        .close-modal {{
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            cursor: pointer;
            z-index: 1001;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .filter-bar {{
                flex-direction: column;
                align-items: stretch;
            }}
            
            .filter-stats {{
                margin-left: 0;
                justify-content: center;
            }}
            
            .student-card {{
                grid-template-columns: 1fr;
            }}
            
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* Export buttons */
        .export-btn {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .export-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }}
        
        .export-btn.secondary {{
            background: linear-gradient(135deg, var(--gray-600) 0%, var(--gray-800) 100%);
        }}
        
        .export-btn.secondary:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }}
        
        /* Edit table */
        .edit-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .edit-table th {{
            background: var(--gray-800);
            color: white;
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
            font-size: 0.8rem;
            position: sticky;
            top: 0;
        }}
        
        .edit-table th.student-col {{
            text-align: left;
            min-width: 200px;
        }}
        
        .edit-table td {{
            padding: 8px;
            border-bottom: 1px solid var(--gray-200);
            text-align: center;
        }}
        
        .edit-table td.student-name {{
            text-align: left;
            font-weight: 500;
        }}
        
        .edit-table tr:hover {{
            background: var(--gray-50);
        }}
        
        .edit-table select {{
            padding: 6px 10px;
            border: 2px solid var(--gray-200);
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            background: white;
        }}
        
        .edit-table select:focus {{
            outline: none;
            border-color: var(--primary);
        }}
        
        .edit-table select.modified {{
            border-color: var(--warning);
            background: #fef3c7;
        }}
        
        .edit-table .class-badge {{
            display: inline-block;
            background: var(--primary);
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
            margin-left: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Exam Results Dashboard</h1>
            <p class="subtitle">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </header>
        
        <!-- Filter Bar -->
        <div class="filter-bar">
            <div class="filter-group">
                <label>Class</label>
                <select id="filterClass" onchange="applyAllFilters()">
                    <option value="all">All Classes</option>
"""
    
    for class_name in all_class_data.keys():
        html_content += f'                    <option value="{class_name}">{class_name}</option>\n'
    
    html_content += """
                </select>
            </div>
            
            <div class="filter-group">
                <label>Score Range</label>
                <select id="filterScore" onchange="applyAllFilters()">
                    <option value="all">All Scores</option>
                    <option value="high">High (≥70%)</option>
                    <option value="medium">Medium (50-69%)</option>
                    <option value="low">Low (&lt;50%)</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Search Student</label>
                <input type="text" id="searchStudent" placeholder="Type name..." oninput="applyAllFilters()">
            </div>
            
            <div class="filter-stats">
                <div class="filter-stat">
                    <div class="filter-stat-value" id="visibleCount">0</div>
                    <div class="filter-stat-label">Showing</div>
                </div>
                <div class="filter-stat">
                    <div class="filter-stat-value" id="totalCount">0</div>
                    <div class="filter-stat-label">Total</div>
                </div>
            </div>
        </div>
        
        <!-- Tabs -->
        <div class="tabs">
            <button class="tab active" onclick="showSection('overview')">📈 Overview</button>
            <button class="tab" onclick="showSection('students')">👥 Students</button>
            <button class="tab" onclick="showSection('questions')">❓ Questions</button>
            <button class="tab" onclick="showSection('tags')">🏷️ Tags</button>
            <button class="tab" onclick="showSection('edit')">✏️ Edit Results</button>
        </div>
        
        <!-- Overview Section -->
        <div id="overview" class="section active">
            <h2 class="section-title">📈 Performance Overview</h2>
            
            <div class="stats-grid" id="overviewStats"></div>
            
            <div class="charts-grid">
                <div class="chart-card">
                    <h3>Score Distribution by Class</h3>
                    <canvas id="scoreDistChart"></canvas>
                </div>
                <div class="chart-card">
                    <h3>Class Average Comparison</h3>
                    <canvas id="classAvgChart"></canvas>
                </div>
                <div class="chart-card">
                    <h3>Question Accuracy Heatmap</h3>
                    <canvas id="questionAccChart"></canvas>
                </div>
                <div class="chart-card">
                    <h3>Tag Performance</h3>
                    <canvas id="tagChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Students Section -->
        <div id="students" class="section">
            <h2 class="section-title">👥 Student Results</h2>
            <div class="students-grid" id="studentsGrid"></div>
        </div>
        
        <!-- Questions Section -->
        <div id="questions" class="section">
            <h2 class="section-title">❓ Question Analysis</h2>
            <div id="questionsContent"></div>
        </div>
        
        <!-- Tags Section -->
        <div id="tags" class="section">
            <h2 class="section-title">🏷️ Tag Analysis</h2>
            <div id="tagsContent"></div>
        </div>
        
        <!-- Edit Results Section -->
        <div id="edit" class="section">
            <h2 class="section-title">✏️ Edit Results</h2>
            <p style="margin-bottom: 20px; color: var(--gray-600);">Manually correct detection errors. Changes are applied locally and can be exported.</p>
            <div style="margin-bottom: 20px; display: flex; gap: 10px;">
                <button onclick="exportResults()" class="export-btn">💾 Export to CSV</button>
                <button onclick="exportResultsJSON()" class="export-btn secondary">📋 Export to JSON</button>
                <span id="editStatus" style="margin-left: auto; color: var(--success); font-weight: 500;"></span>
            </div>
            <div id="editContent" style="overflow-x: auto;"></div>
        </div>
    </div>
    
    <!-- Modal -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close-modal">&times;</span>
        <img class="modal-content" id="modalImg">
    </div>
    
    <script>
        // Data
        const allStudents = """ + json.dumps(all_students_json) + """;
        const chartData = """ + json.dumps(chart_data) + """;
        const classData = """ + json.dumps({k: {'total_questions': v['total_questions'], 'question_stats': {str(q): s for q, s in v['question_stats'].items()}, 'tag_stats': v['tag_stats']} for k, v in all_class_data.items()}) + """;
        const answerKeyData = """ + json.dumps({k: v.get('answer_key', {}) for k, v in all_class_data.items()}) + """;
        const studentAnswersData = """ + json.dumps({s['name']: {'class': s['class'], 'answers': s.get('answers', {})} for s in all_students_json}) + """;
        
        // Chart instances for re-rendering
        let chartInstances = {};
        
        // Editable data (copy of student answers)
        let editableData = JSON.parse(JSON.stringify(allStudents.map(s => ({
            name: s.name,
            class: s.class,
            answers: {...s.answers},
            originalAnswers: {...s.answers}
        }))));
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            renderStudents();
            renderOverviewStats();
            renderCharts();
            renderQuestions();
            renderTags();
            renderEditTable();
            applyAllFilters();
        });
        
        function applyAllFilters() {
            filterStudents();
            filterQuestions();
            filterTags();
            filterEditTable();
            renderOverviewStats();
            renderCharts();
        }
        
        function showSection(sectionId) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(sectionId).classList.add('active');
            event.target.classList.add('active');
            
            // Re-apply filters when switching tabs
            filterStudents();
            filterQuestions();
            filterTags();
        }
        
        function filterStudents() {
            const classFilter = document.getElementById('filterClass').value;
            const scoreFilter = document.getElementById('filterScore').value;
            const searchFilter = document.getElementById('searchStudent').value.toLowerCase();
            
            let visible = 0;
            document.querySelectorAll('.student-card').forEach(card => {
                const studentClass = card.dataset.class;
                const percentage = parseFloat(card.dataset.percentage);
                const name = card.dataset.name.toLowerCase();
                
                let show = true;
                
                if (classFilter !== 'all' && studentClass !== classFilter) show = false;
                if (scoreFilter === 'high' && percentage < 70) show = false;
                if (scoreFilter === 'medium' && (percentage < 50 || percentage >= 70)) show = false;
                if (scoreFilter === 'low' && percentage >= 50) show = false;
                if (searchFilter && !name.includes(searchFilter)) show = false;
                
                card.style.display = show ? 'grid' : 'none';
                if (show) visible++;
            });
            
            document.getElementById('visibleCount').textContent = visible;
            document.getElementById('totalCount').textContent = allStudents.length;
        }
        
        function filterQuestions() {
            const classFilter = document.getElementById('filterClass').value;
            document.querySelectorAll('#questionsContent .class-section').forEach(section => {
                const sectionClass = section.dataset.class;
                if (classFilter === 'all' || sectionClass === classFilter) {
                    section.style.display = 'block';
                } else {
                    section.style.display = 'none';
                }
            });
        }
        
        function filterTags() {
            const classFilter = document.getElementById('filterClass').value;
            document.querySelectorAll('#tagsContent .class-section').forEach(section => {
                const sectionClass = section.dataset.class;
                if (classFilter === 'all' || sectionClass === classFilter) {
                    section.style.display = 'block';
                } else {
                    section.style.display = 'none';
                }
            });
        }
        
        function toggleStudentCard(card) {
            // Close any other expanded cards
            document.querySelectorAll('.student-card.expanded').forEach(c => {
                if (c !== card) c.classList.remove('expanded');
            });
            // Toggle this card
            card.classList.toggle('expanded');
        }
        
        function renderStudents() {
            const grid = document.getElementById('studentsGrid');
            
            // Sort by percentage descending
            const sorted = [...allStudents].sort((a, b) => b.percentage - a.percentage);
            
            grid.innerHTML = sorted.map(s => {
                const scoreClass = s.percentage >= 70 ? 'high' : s.percentage >= 50 ? 'medium' : 'low';
                const debugImg = `student_exams/${s.class}/debug_output/${s.name}_debug.jpg`;
                const otsuImg = `student_exams/${s.class}/debug_output/${s.name}_otsu.jpg`;
                const missedPreview = s.missed_tags.length > 0 ? s.missed_tags.slice(0, 3).join(', ') + (s.missed_tags.length > 3 ? '...' : '') : 'None';
                const missedTagsHtml = s.missed_tags.length > 0 
                    ? s.missed_tags.map(t => `<span class="missed-tag">${t}</span>`).join('')
                    : '<span class="no-missed">✓ No missed tags - Perfect understanding!</span>';
                
                return `
                    <div class="student-card" data-class="${s.class}" data-percentage="${s.percentage}" data-name="${s.name}" onclick="toggleStudentCard(this)">
                        <div class="student-main">
                            <div class="student-info">
                                <h4>${s.name}</h4>
                                <span class="class-badge">${s.class}</span>
                                <div class="tags"><strong>Missed:</strong> ${missedPreview}</div>
                            </div>
                            <div class="student-score">
                                <div class="score-circle ${scoreClass}">${s.percentage}%</div>
                                <div class="score-label">${s.score}/${s.total}</div>
                            </div>
                            <div class="student-images">
                                <img src="${debugImg}" onclick="event.stopPropagation(); openModal('${debugImg}')" alt="Debug" onerror="this.style.display='none'">
                                <img src="${otsuImg}" onclick="event.stopPropagation(); openModal('${otsuImg}')" alt="Otsu" onerror="this.style.display='none'">
                            </div>
                        </div>
                        <div class="student-details">
                            <strong>📌 All Missed Tags (${s.missed_tags.length}):</strong>
                            <div class="missed-tags-full">
                                ${missedTagsHtml}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function getFilteredStudents() {
            const classFilter = document.getElementById('filterClass').value;
            const scoreFilter = document.getElementById('filterScore').value;
            const searchFilter = document.getElementById('searchStudent').value.toLowerCase();
            
            return allStudents.filter(s => {
                if (classFilter !== 'all' && s.class !== classFilter) return false;
                if (scoreFilter === 'high' && s.percentage < 70) return false;
                if (scoreFilter === 'medium' && (s.percentage < 50 || s.percentage >= 70)) return false;
                if (scoreFilter === 'low' && s.percentage >= 50) return false;
                if (searchFilter && !s.name.toLowerCase().includes(searchFilter)) return false;
                return true;
            });
        }
        
        function renderOverviewStats() {
            const filtered = getFilteredStudents();
            const totalStudents = filtered.length;
            const avgScore = filtered.reduce((sum, s) => sum + s.percentage, 0) / totalStudents || 0;
            const highPerformers = filtered.filter(s => s.percentage >= 70).length;
            const lowPerformers = filtered.filter(s => s.percentage < 50).length;
            
            document.getElementById('overviewStats').innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${totalStudents}</div>
                    <div class="stat-label">Filtered Students</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-value">${avgScore.toFixed(1)}%</div>
                    <div class="stat-label">Average Score</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-value">${highPerformers}</div>
                    <div class="stat-label">High Performers (≥70%)</div>
                </div>
                <div class="stat-card danger">
                    <div class="stat-value">${lowPerformers}</div>
                    <div class="stat-label">Need Support (&lt;50%)</div>
                </div>
            `;
        }
        
        function renderCharts() {
            const classFilter = document.getElementById('filterClass').value;
            const filtered = getFilteredStudents();
            
            // Destroy existing charts
            Object.values(chartInstances).forEach(chart => chart.destroy());
            chartInstances = {};
            
            // Determine which classes to show
            const classes = classFilter === 'all' ? Object.keys(chartData) : [classFilter];
            const colors = ['#4f46e5', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
            
            // Score Distribution
            chartInstances.scoreDist = new Chart(document.getElementById('scoreDistChart'), {
                type: 'bar',
                data: {
                    labels: ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'],
                    datasets: classes.map((c, i) => {
                        const classStudents = filtered.filter(s => s.class === c);
                        const bins = [0, 0, 0, 0, 0];
                        classStudents.forEach(s => {
                            const pct = s.percentage;
                            if (pct <= 20) bins[0]++;
                            else if (pct <= 40) bins[1]++;
                            else if (pct <= 60) bins[2]++;
                            else if (pct <= 80) bins[3]++;
                            else bins[4]++;
                        });
                        return {
                            label: c,
                            data: bins,
                            backgroundColor: colors[i % colors.length]
                        };
                    })
                },
                options: {
                    responsive: true,
                    plugins: { legend: { position: 'top' } },
                    scales: { y: { beginAtZero: true } }
                }
            });
            
            // Class Average (filtered)
            const classAvgs = classes.map(c => {
                const classStudents = filtered.filter(s => s.class === c);
                return classStudents.length > 0 
                    ? (classStudents.reduce((sum, s) => sum + s.percentage, 0) / classStudents.length).toFixed(1)
                    : 0;
            });
            
            chartInstances.classAvg = new Chart(document.getElementById('classAvgChart'), {
                type: 'doughnut',
                data: {
                    labels: classes,
                    datasets: [{
                        data: classAvgs,
                        backgroundColor: colors
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'right' },
                        tooltip: {
                            callbacks: {
                                label: ctx => `${ctx.label}: ${ctx.raw}% avg`
                            }
                        }
                    }
                }
            });
            
            // Question Accuracy (for selected class or first class)
            const targetClass = classes[0];
            const qAcc = chartData[targetClass].question_accuracy;
            const qLabels = Object.keys(qAcc).sort((a, b) => parseInt(a) - parseInt(b));
            
            chartInstances.questionAcc = new Chart(document.getElementById('questionAccChart'), {
                type: 'bar',
                data: {
                    labels: qLabels.map(q => 'Q' + q),
                    datasets: [{
                        label: 'Accuracy %',
                        data: qLabels.map(q => qAcc[q].toFixed(1)),
                        backgroundColor: qLabels.map(q => qAcc[q] >= 70 ? '#10b981' : qAcc[q] >= 50 ? '#f59e0b' : '#ef4444')
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true, max: 100 } }
                }
            });
            
            // Tag Performance (combined across filtered classes)
            const allTags = {};
            classes.forEach(c => {
                Object.entries(chartData[c].tag_accuracy).forEach(([tag, acc]) => {
                    if (!allTags[tag]) allTags[tag] = [];
                    allTags[tag].push(acc);
                });
            });
            const tagLabels = Object.keys(allTags);
            const tagAvgs = tagLabels.map(t => allTags[t].reduce((a, b) => a + b, 0) / allTags[t].length);
            
            chartInstances.tagChart = new Chart(document.getElementById('tagChart'), {
                type: 'radar',
                data: {
                    labels: tagLabels,
                    datasets: [{
                        label: 'Accuracy %',
                        data: tagAvgs,
                        backgroundColor: 'rgba(79, 70, 229, 0.2)',
                        borderColor: '#4f46e5',
                        pointBackgroundColor: '#4f46e5'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        r: { beginAtZero: true, max: 100 }
                    }
                }
            });
        }
        
        function renderQuestions() {
            let html = '';
            Object.entries(classData).forEach(([className, data]) => {
                html += `<div class="class-section" data-class="${className}">`;
                html += `<h3 style="margin: 20px 0 10px; color: #4f46e5;">${className}</h3>`;
                html += `<table><thead><tr><th>Question</th><th>Correct</th><th>Incorrect</th><th>Accuracy</th><th>Visual</th></tr></thead><tbody>`;
                
                Object.entries(data.question_stats).sort((a, b) => parseInt(a[0]) - parseInt(b[0])).forEach(([q, stats]) => {
                    const total = stats.correct + stats.incorrect;
                    const acc = total > 0 ? (stats.correct / total * 100) : 0;
                    const accClass = acc >= 70 ? 'high' : acc >= 50 ? 'medium' : 'low';
                    html += `
                        <tr>
                            <td><strong>Q${q}</strong></td>
                            <td>${stats.correct}</td>
                            <td>${stats.incorrect}</td>
                            <td>${acc.toFixed(1)}%</td>
                            <td><div class="accuracy-bar"><div class="accuracy-fill ${accClass}" style="width: ${acc}%"></div></div></td>
                        </tr>
                    `;
                });
                html += '</tbody></table></div>';
            });
            document.getElementById('questionsContent').innerHTML = html;
        }
        
        function renderTags() {
            let html = '';
            Object.entries(classData).forEach(([className, data]) => {
                if (Object.keys(data.tag_stats).length === 0) return;
                
                html += `<div class="class-section" data-class="${className}">`;
                html += `<h3 style="margin: 20px 0 10px; color: #4f46e5;">${className}</h3>`;
                html += `<table><thead><tr><th>Tag</th><th>Correct</th><th>Incorrect</th><th>Accuracy</th><th>Visual</th></tr></thead><tbody>`;
                
                Object.entries(data.tag_stats).sort((a, b) => a[0].localeCompare(b[0])).forEach(([tag, stats]) => {
                    const total = stats.correct + stats.incorrect;
                    const acc = total > 0 ? (stats.correct / total * 100) : 0;
                    const accClass = acc >= 70 ? 'high' : acc >= 50 ? 'medium' : 'low';
                    html += `
                        <tr>
                            <td><strong>${tag}</strong></td>
                            <td>${stats.correct}</td>
                            <td>${stats.incorrect}</td>
                            <td>${acc.toFixed(1)}%</td>
                            <td><div class="accuracy-bar"><div class="accuracy-fill ${accClass}" style="width: ${acc}%"></div></div></td>
                        </tr>
                    `;
                });
                html += '</tbody></table></div>';
            });
            document.getElementById('tagsContent').innerHTML = html || '<p>No tag data available.</p>';
        }
        
        function renderEditTable() {
            const totalQuestions = 25;
            let html = '<table class="edit-table"><thead><tr><th class="student-col">Student</th>';
            
            for (let q = 1; q <= totalQuestions; q++) {
                html += `<th>Q${q}</th>`;
            }
            html += '</tr></thead><tbody>';
            
            // Sort by class then name
            const sorted = [...allStudents].sort((a, b) => {
                if (a.class !== b.class) return a.class.localeCompare(b.class);
                return a.name.localeCompare(b.name);
            });
            
            sorted.forEach((student, idx) => {
                html += `<tr data-class="${student.class}" data-name="${student.name}">`;
                html += `<td class="student-name">${student.name}<span class="class-badge">${student.class}</span></td>`;
                
                for (let q = 1; q <= totalQuestions; q++) {
                    const currentAnswer = student.answers ? student.answers[q] || '' : '';
                    html += `<td>
                        <select onchange="updateAnswer('${student.name}', ${q}, this.value, this)" data-original="${currentAnswer}">
                            <option value="" ${currentAnswer === '' ? 'selected' : ''}>-</option>
                            <option value="A" ${currentAnswer === 'A' ? 'selected' : ''}>A</option>
                            <option value="B" ${currentAnswer === 'B' ? 'selected' : ''}>B</option>
                            <option value="C" ${currentAnswer === 'C' ? 'selected' : ''}>C</option>
                            <option value="D" ${currentAnswer === 'D' ? 'selected' : ''}>D</option>
                            <option value="E" ${currentAnswer === 'E' ? 'selected' : ''}>E</option>
                        </select>
                    </td>`;
                }
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            document.getElementById('editContent').innerHTML = html;
        }
        
        function filterEditTable() {
            const classFilter = document.getElementById('filterClass').value;
            const searchFilter = document.getElementById('searchStudent').value.toLowerCase();
            
            document.querySelectorAll('#editContent tbody tr').forEach(row => {
                const rowClass = row.dataset.class;
                const name = row.dataset.name.toLowerCase();
                
                let show = true;
                if (classFilter !== 'all' && rowClass !== classFilter) show = false;
                if (searchFilter && !name.includes(searchFilter)) show = false;
                
                row.style.display = show ? '' : 'none';
            });
        }
        
        function updateAnswer(studentName, questionNum, newValue, selectEl) {
            // Find and update the student in allStudents
            const student = allStudents.find(s => s.name === studentName);
            if (student) {
                if (!student.answers) student.answers = {};
                student.answers[questionNum] = newValue;
                
                // Mark as modified if different from original
                const original = selectEl.dataset.original;
                if (newValue !== original) {
                    selectEl.classList.add('modified');
                } else {
                    selectEl.classList.remove('modified');
                }
                
                document.getElementById('editStatus').textContent = 'Changes pending...';
            }
        }
        
        function exportResults() {
            let csv = 'Name,Class,Score,Total,Percentage';
            for (let q = 1; q <= 25; q++) csv += `,Q${q}`;
            csv += '\\n';
            
            allStudents.forEach(s => {
                const answers = s.answers || {};
                csv += `"${s.name}","${s.class}",${s.score},${s.total},${s.percentage}`;
                for (let q = 1; q <= 25; q++) {
                    csv += `,${answers[q] || ''}`;
                }
                csv += '\\n';
            });
            
            downloadFile(csv, 'exam_results_edited.csv', 'text/csv');
            document.getElementById('editStatus').textContent = '✅ CSV exported!';
        }
        
        function exportResultsJSON() {
            const data = allStudents.map(s => ({
                name: s.name,
                class: s.class,
                score: s.score,
                total: s.total,
                percentage: s.percentage,
                answers: s.answers || {},
                missed_tags: s.missed_tags
            }));
            
            const json = JSON.stringify(data, null, 2);
            downloadFile(json, 'exam_results_edited.json', 'application/json');
            document.getElementById('editStatus').textContent = '✅ JSON exported!';
        }
        
        function downloadFile(content, filename, mimeType) {
            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
        
        function openModal(imgSrc) {
            event.stopPropagation();
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImg').src = imgSrc;
        }
        
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Dashboard generated: {output_file}")

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Folder '{ROOT_DIR}' not found.")
        return

    # Store all class data for dashboard
    all_class_data = {}

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
        question_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
        tag_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
        student_data_list = []
        
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
                    question_stats[q_num]['correct'] += 1
                    
                    # Track tag statistics (correct)
                    if q_num in question_tags:
                        for tag in question_tags[q_num]:
                            tag_stats[tag]['correct'] += 1
                else:
                    question_stats[q_num]['incorrect'] += 1
                    
                    # Track missed tags
                    if q_num in question_tags:
                        for tag in question_tags[q_num]:
                            missed_tags.add(tag)
                            tag_stats[tag]['incorrect'] += 1
                    
                # Save answer detail? (Optional, helps debugging)
                student_data[f'Q{q_num}'] = student_ans

            student_data['Total Score'] = score
            student_data['Missed Tags'] = ', '.join(sorted(missed_tags)) if missed_tags else 'None'
            class_results.append(student_data)
            
            # Store for dashboard (include answers for edit functionality)
            student_data_list.append({
                'name': s_name,
                'score': score,
                'missed_tags': sorted(missed_tags),
                'answers': {int(k): v for k, v in answers.items()}
            })

        # 7. Save Class CSV
        if class_results:
            df = pd.DataFrame(class_results)
            
            # Reorder columns: Name, Score, Missed Tags, Q1, Q2...
            cols = ['Name', 'Total Score', 'Missed Tags'] + [c for c in df.columns if c not in ['Name', 'Total Score', 'Missed Tags']]
            df = df[cols]
            
            output_filename = f"results_{class_folder}.csv"
            df.to_csv(output_filename, index=False)
            print(f"   [Done] Saved {output_filename}")
            
            # Store class data for dashboard
            all_class_data[class_folder] = {
                'students': student_data_list,
                'question_stats': dict(question_stats),
                'tag_stats': dict(tag_stats),
                'total_questions': len(class_key)
            }
        else:
            print(f"   [Info] No students found in {class_folder}")

    # 8. Generate Dashboard
    if all_class_data:
        generate_dashboard(all_class_data)
    else:
        print("\n[Info] No data to generate dashboard.")

if __name__ == "__main__":
    main()