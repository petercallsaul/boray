import os
import sys
import csv
import numpy as np

def pre_process_landmarks(coords):
    """
    Normalizes 21 landmark pairs [x, y] exactly like app.py:
    1. Translate coordinates so wrist (landmark 0) is at (0, 0)
    2. Normalize all values by the maximum absolute coordinate value
    """
    arr = np.array(coords, dtype=np.float32).reshape(-1, 2)
    arr -= arr[0] # wrist translation
    flat = arr.flatten()
    max_val = np.max(np.abs(flat))
    if max_val > 0:
        flat /= max_val
    return flat.tolist()

def convert_dataset(input_csv_path, output_keypoint_path='model/keypoint_classifier/keypoint.csv',
                    output_label_path='model/keypoint_classifier/keypoint_classifier_label.csv'):
    if not os.path.exists(input_csv_path):
        print(f"Error: Could not find '{input_csv_path}'.")
        print("Please place the downloaded Arabic Sign Language CSV in the SignAI folder or specify the path.")
        return

    print(f"Reading dataset from: {input_csv_path}")

    # Arabic letter name mapping dictionary for ArSL-31
    ARABIC_MAP = {
        'Alef': 'أ', 'Ba2': 'ب', 'Ta2': 'ت', 'Tha2': 'ث', 'Jim': 'ج',
        '7a2': 'ح', 'Kha2': 'خ', 'Dal': 'د', 'Thal': 'ذ', 'Ra2': 'ر',
        'Zayn': 'ز', 'Sin': 'س', 'Chin': 'ش', 'SSad': 'ص', 'DDad': 'ض',
        'TTa2': 'ط', 'TTha2': 'ظ', '3ayn': 'ع', 'Ghayn': 'غ', 'Fa2': 'ف',
        '9af': 'ق', 'Kaf': 'ك', 'Lam': 'ل', 'Mim': 'م', 'Noon': 'ن',
        'Ha2': 'هـ', 'Waw': 'و', 'Ya2': 'ي',
        'Space': 'مسافة', 'Delete': 'مسح', 'Finish': 'نقطة',
        # Alternative aliases
        'Sign_Alef': 'أ', 'Sign_Ba2': 'ب', 'Sign_Ta2': 'ت', 'Sign_Tha2': 'ث',
        'Sign_Jeem': 'ج', 'Sign_7a2': 'ح', 'Sign_Kha2': 'خ', 'Sign_Dal': 'د',
        'Sign_Thal': 'ذ', 'Sign_Ra2': 'ر', 'Sign_Zay': 'ز', 'Sign_Seen': 'س',
        'Sign_Sheen': 'ش', 'Sign_Sad': 'ص', 'Sign_Dhad': 'ض', 'Sign_Tah': 'ط',
        'Sign_Zah': 'ظ', 'Sign_3ain': 'ع', 'Sign_Ghain': 'غ', 'Sign_Fa2': 'ف',
        'Sign_Qaf': 'ق', 'Sign_Kaf': 'ك', 'Sign_Lam': 'ل', 'Sign_Meem': 'م',
        'Sign_Noon': 'ن', 'Sign_Ha2': 'هـ', 'Sign_Waw': 'و', 'Sign_Ya2': 'ي',
        'Sign_Space': 'مسافة', 'Sign_Delete': 'مسح', 'Sign_Finish': 'نقطة'
    }

    labels_list = []
    label_to_id = {}
    rows_to_write = []

    # Detect delimiter
    with open(input_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        delimiter = ';' if ';' in first_line else ','

    with open(input_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)

        # Detect label column (first column or last column)
        label_col = 0
        if header[-1].lower() in ['label', 'class', 'target']:
            label_col = len(header) - 1

        for row in reader:
            if not row or len(row) < 43:
                continue

            raw_label = row[label_col].strip()
            arabic_char = ARABIC_MAP.get(raw_label, raw_label)

            if arabic_char not in label_to_id:
                label_to_id[arabic_char] = len(labels_list)
                labels_list.append(arabic_char)

            class_id = label_to_id[arabic_char]

            # Extract 42 landmark coordinates (21 points x 2)
            coords = []
            start_idx = 1 if label_col == 0 else 0
            for val in row[start_idx:start_idx + 42]:
                try:
                    coords.append(float(val))
                except ValueError:
                    coords.append(0.0)

            if len(coords) < 42:
                continue

            # Pre-process landmarks according to pipeline
            normalized_42 = pre_process_landmarks(coords[:42])

            # For 1-hand dataset, pad slots 0..41 with 0.0, landmarks in 42..83
            full_row = [class_id] + [0.0] * 42 + normalized_42
            rows_to_write.append(full_row)

    print(f"Loaded {len(rows_to_write)} samples across {len(labels_list)} Arabic classes.")

    # Write keypoint.csv
    os.makedirs(os.path.dirname(output_keypoint_path), exist_ok=True)
    with open(output_keypoint_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows_to_write)
    print(f"Saved keypoints to: {output_keypoint_path}")

    # Write keypoint_classifier_label.csv
    with open(output_label_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        for lbl in labels_list:
            writer.writerow([lbl])
    print(f"Saved labels to: {output_label_path}")
    print("\nDataset conversion complete! Now run: python train_model.py")

if __name__ == '__main__':
    target_csv = sys.argv[1] if len(sys.argv) > 1 else 'ArSL_dataset.csv'
    convert_dataset(target_csv)
