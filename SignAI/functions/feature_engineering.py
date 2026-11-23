import math
import numpy as np

def compute_angle(a, b, c):
    """
    Computes cosine of the angle ABC at vertex B.
    Points a, b, c are 2D coordinates [x, y].
    Returns cosine value in range [-1.0, 1.0].
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.clip(cosine, -1.0, 1.0))

def extract_hand_features(landmarks_coords):
    """
    Extracts 74 rotation-resilient and scale-invariant kinematic features from 21 hand landmarks.
    landmarks_coords: list or array of shape (21, 2) containing (x, y) coordinates.
    Returns: list of 74 float features.
    """
    pts = np.array(landmarks_coords, dtype=np.float32).reshape(21, 2)
    wrist = pts[0]

    # 1. Palm Scale Factor: distance from wrist (0) to middle finger MCP (9)
    palm_scale = float(np.linalg.norm(pts[9] - wrist))
    if palm_scale < 1e-5:
        palm_scale = 1.0

    # 2. Wrist-relative, scale-normalized coordinates (42 features)
    rel_coords = (pts - wrist) / palm_scale
    flat_rel = rel_coords.flatten().tolist()

    # 3. Radial distances from wrist to fingertips (5 features)
    fingertip_indices = [4, 8, 12, 16, 20]
    radial_dists = [
        float(np.linalg.norm(pts[idx] - wrist) / palm_scale)
        for idx in fingertip_indices
    ]

    # 4. Critical inter-fingertip and aperture distances (8 features)
    # Normalized by palm scale
    p_thumb = pts[4]
    p_index = pts[8]
    p_mid = pts[12]
    p_ring = pts[16]
    p_pinky = pts[20]

    inter_dists = [
        float(np.linalg.norm(p_thumb - p_index) / palm_scale),  # Pinch / C-shape aperture
        float(np.linalg.norm(p_thumb - p_mid) / palm_scale),
        float(np.linalg.norm(p_thumb - p_ring) / palm_scale),
        float(np.linalg.norm(p_thumb - p_pinky) / palm_scale),
        float(np.linalg.norm(p_index - p_mid) / palm_scale),
        float(np.linalg.norm(p_mid - p_ring) / palm_scale),
        float(np.linalg.norm(p_ring - p_pinky) / palm_scale),
        float(np.linalg.norm(p_index - p_pinky) / palm_scale),  # Laa / Horn span
    ]

    # 5. Joint flexion cosine angles (14 features)
    joint_angles = [
        # Thumb: (1, 2, 3) and (2, 3, 4)
        compute_angle(pts[1], pts[2], pts[3]),
        compute_angle(pts[2], pts[3], pts[4]),
        # Index: (0, 5, 6), (5, 6, 7), (6, 7, 8)
        compute_angle(pts[0], pts[5], pts[6]),
        compute_angle(pts[5], pts[6], pts[7]),
        compute_angle(pts[6], pts[7], pts[8]),
        # Middle: (0, 9, 10), (9, 10, 11), (10, 11, 12)
        compute_angle(pts[0], pts[9], pts[10]),
        compute_angle(pts[9], pts[10], pts[11]),
        compute_angle(pts[10], pts[11], pts[12]),
        # Ring: (0, 13, 14), (13, 14, 15), (14, 15, 16)
        compute_angle(pts[0], pts[13], pts[14]),
        compute_angle(pts[13], pts[14], pts[15]),
        compute_angle(pts[14], pts[15], pts[16]),
        # Pinky: (0, 17, 18), (17, 18, 19), (18, 19, 20)
        compute_angle(pts[0], pts[17], pts[18]),
        compute_angle(pts[17], pts[18], pts[19]),
        compute_angle(pts[18], pts[19], pts[20]),
    ]

    # 6. Finger extension deltas: fingertip radial distance minus MCP radial distance (5 features)
    mcp_indices = [2, 5, 9, 13, 17]
    extension_deltas = [
        float((np.linalg.norm(pts[tip] - wrist) - np.linalg.norm(pts[mcp] - wrist)) / palm_scale)
        for tip, mcp in zip(fingertip_indices, mcp_indices)
    ]

    # Total: 42 + 5 + 8 + 14 + 5 = 74 features
    features = flat_rel + radial_dists + inter_dists + joint_angles + extension_deltas
    return features


def extract_dual_hand_features(hand_features_list):
    """
    Constructs a fixed 148-dimensional feature vector for 1 or 2 hands.
    If 2 hands are present: Hand1 (74) + Hand2 (74).
    If 1 hand is present: Zeros (74) + Hand1 (74).
    If 0 hands are present: Zeros (148).
    """
    if len(hand_features_list) >= 2:
        return hand_features_list[0] + hand_features_list[1]
    elif len(hand_features_list) == 1:
        return [0.0] * 74 + hand_features_list[0]
    else:
        return [0.0] * 148


def augment_landmarks(landmarks_coords, max_angle_deg=15.0, scale_range=(0.90, 1.10), noise_std=0.008):
    """
    Applies synthetic data augmentation on 21 landmark 2D coordinates:
    - Random 2D planar rotation [-max_angle, +max_angle] around wrist
    - Random uniform scaling
    - Additive Gaussian noise
    """
    pts = np.array(landmarks_coords, dtype=np.float32).reshape(21, 2).copy()
    wrist = pts[0].copy()

    # 1. Random rotation around wrist
    angle_rad = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    centered = pts - wrist
    rotated = np.dot(centered, rot_matrix)

    # 2. Random scaling
    scale = np.random.uniform(scale_range[0], scale_range[1])
    scaled = rotated * scale

    # 3. Additive Gaussian noise
    noise = np.random.normal(0, noise_std, size=scaled.shape).astype(np.float32)
    # Keep wrist at origin without jitter
    noise[0] = 0.0

    augmented = scaled + noise + wrist
    return augmented.flatten().tolist()


def mirror_landmarks(landmarks_coords):
    """
    Mirrors landmarks along the vertical axis for left-to-right hand invariance.
    """
    pts = np.array(landmarks_coords, dtype=np.float32).reshape(21, 2).copy()
    wrist_x = pts[0, 0]
    pts[:, 0] = 2.0 * wrist_x - pts[:, 0]
    return pts.flatten().tolist()
