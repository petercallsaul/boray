import os
import csv
import numpy as np

def generate_hand_landmarks(pose_type):
    """
    Generates 21 landmark (x, y) coordinates for the 8 dental signs.
    Wrist is at (0, 0).
    """
    pts = np.zeros((21, 2), dtype=np.float32)
    # 0: wrist
    pts[0] = [0.0, 0.0]
    # Thumb (1-4)
    pts[1] = [-0.15, -0.15]
    pts[2] = [-0.25, -0.30]
    pts[3] = [-0.30, -0.42]
    pts[4] = [-0.35, -0.55]
    # Index (5-8)
    pts[5] = [-0.10, -0.40]
    pts[6] = [-0.12, -0.60]
    pts[7] = [-0.13, -0.75]
    pts[8] = [-0.14, -0.90]
    # Middle (9-12)
    pts[9] = [0.00, -0.42]
    pts[10] = [0.00, -0.63]
    pts[11] = [0.00, -0.79]
    pts[12] = [0.00, -0.95]
    # Ring (13-16)
    pts[13] = [0.10, -0.40]
    pts[14] = [0.12, -0.60]
    pts[15] = [0.13, -0.75]
    pts[16] = [0.14, -0.89]
    # Pinky (17-20)
    pts[17] = [0.20, -0.35]
    pts[18] = [0.23, -0.52]
    pts[19] = [0.25, -0.65]
    pts[20] = [0.27, -0.77]

    def curl_finger(mcp, pip, dip, tip):
        base = pts[mcp]
        pts[pip] = [base[0] * 1.1, base[1] - 0.20]
        pts[dip] = [base[0] * 1.15, base[1] - 0.12]
        pts[tip] = [base[0] * 1.05, base[1] - 0.05]

    def curl_thumb():
        pts[1] = [-0.10, -0.15]
        pts[2] = [-0.15, -0.22]
        pts[3] = [-0.10, -0.28]
        pts[4] = [-0.05, -0.30]

    if pose_type == 0:  # أشعر بألم (Fist - All fingers curled)
        curl_thumb()
        curl_finger(5, 6, 7, 8)
        curl_finger(9, 10, 11, 12)
        curl_finger(13, 14, 15, 16)
        curl_finger(17, 18, 19, 20)

    elif pose_type == 1:  # انا تمام (OK Sign - Thumb and Index tips touch)
        pts[1] = [-0.12, -0.15]
        pts[2] = [-0.18, -0.28]
        pts[3] = [-0.18, -0.42]
        pts[4] = [-0.12, -0.55]
        pts[5] = [-0.08, -0.38]
        pts[6] = [-0.14, -0.48]
        pts[7] = [-0.14, -0.55]
        pts[8] = [-0.12, -0.55]
        # Middle, Ring, Pinky remain extended straight up

    elif pose_type == 2:  # قف (Stop - Open palm, 5 fingers upright)
        pts[4] = [-0.35, -0.55]
        pts[8] = [-0.14, -0.90]
        pts[12] = [0.00, -0.95]
        pts[16] = [0.14, -0.89]
        pts[20] = [0.27, -0.77]

    elif pose_type == 3:  # ألم خفيف (Mild pain - Small pinch, index and thumb close, others curled)
        pts[1] = [-0.12, -0.18]
        pts[2] = [-0.20, -0.32]
        pts[3] = [-0.20, -0.45]
        pts[4] = [-0.10, -0.52]
        pts[5] = [-0.05, -0.35]
        pts[6] = [-0.08, -0.50]
        pts[7] = [-0.09, -0.58]
        pts[8] = [-0.10, -0.62]
        curl_finger(9, 10, 11, 12)
        curl_finger(13, 14, 15, 16)
        curl_finger(17, 18, 19, 20)

    elif pose_type == 4:  # ألم قوي (Severe pain - Wide C with 3 fingers curled into palm)
        pts[1] = [-0.15, -0.15]
        pts[2] = [-0.25, -0.22]
        pts[3] = [-0.32, -0.28]
        pts[4] = [-0.38, -0.32]  # Thumb extended down-left
        pts[5] = [-0.05, -0.35]
        pts[6] = [-0.06, -0.55]
        pts[7] = [-0.08, -0.72]
        pts[8] = [-0.18, -0.75]  # Index curved at top, wide gap from thumb
        curl_finger(9, 10, 11, 12)
        curl_finger(13, 14, 15, 16)
        curl_finger(17, 18, 19, 20)

    elif pose_type == 5:  # اريد أن أحك وجهي (Pinch cluster - All 5 tips meeting)
        pts[1] = [-0.10, -0.18]; pts[2] = [-0.12, -0.32]; pts[3] = [-0.08, -0.45]; pts[4] = [-0.02, -0.60]
        pts[5] = [-0.08, -0.35]; pts[6] = [-0.06, -0.48]; pts[7] = [-0.04, -0.56]; pts[8] = [-0.02, -0.62]
        pts[9] = [0.00, -0.38];  pts[10] = [0.00, -0.50];  pts[11] = [0.00, -0.58];  pts[12] = [0.00, -0.63]
        pts[13] = [0.08, -0.35]; pts[14] = [0.06, -0.48]; pts[15] = [0.04, -0.56]; pts[16] = [0.02, -0.62]
        pts[17] = [0.15, -0.30]; pts[18] = [0.10, -0.45]; pts[19] = [0.06, -0.55]; pts[20] = [0.02, -0.60]

    elif pose_type == 6:  # غسل الفم (Index pointing up, all others curled)
        curl_thumb()
        pts[5] = [-0.05, -0.40]
        pts[6] = [-0.07, -0.60]
        pts[7] = [-0.08, -0.78]
        pts[8] = [-0.09, -0.95]
        curl_finger(9, 10, 11, 12)
        curl_finger(13, 14, 15, 16)
        curl_finger(17, 18, 19, 20)

    elif pose_type == 7:  # اريد أن اتحدث (Talking mouth - Flat fingers on top, thumb below)
        pts[1] = [-0.05, -0.15]
        pts[2] = [0.05, -0.18]
        pts[3] = [0.15, -0.18]
        pts[4] = [0.25, -0.18]
        pts[5] = [-0.05, -0.40]; pts[6] = [0.08, -0.42]; pts[7] = [0.20, -0.42]; pts[8] = [0.32, -0.42]
        pts[9] = [-0.02, -0.40]; pts[10] = [0.10, -0.42]; pts[11] = [0.22, -0.42]; pts[12] = [0.33, -0.42]
        pts[13] = [0.02, -0.38]; pts[14] = [0.12, -0.40]; pts[15] = [0.22, -0.40]; pts[16] = [0.32, -0.40]
        pts[17] = [0.05, -0.35]; pts[18] = [0.14, -0.37]; pts[19] = [0.22, -0.37]; pts[20] = [0.30, -0.37]

    # Normalize landmarks exactly like pre_process_landmarks_normalized
    pts -= pts[0]
    flat = pts.flatten()
    max_val = np.max(np.abs(flat))
    if max_val > 0:
        flat /= max_val
    return flat

def main():
    csv_path = 'model/word_classifier/word_keypoint.csv'
    os.makedirs('model/word_classifier', exist_ok=True)
    rows = []
    np.random.seed(42)

    # 60 samples per class = 480 samples across 8 dental signs
    for class_id in range(8):
        for _ in range(60):
            base_feats = generate_hand_landmarks(class_id)
            noise = np.random.normal(0, 0.03, size=base_feats.shape).astype(np.float32)
            sample_feats = base_feats + noise
            m = np.max(np.abs(sample_feats))
            if m > 0:
                sample_feats /= m
            row = [class_id] + [0.0] * 42 + sample_feats.tolist()
            rows.append(row)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Generated {len(rows)} baseline samples across 8 dental signs in: {csv_path}")

if __name__ == '__main__':
    main()
