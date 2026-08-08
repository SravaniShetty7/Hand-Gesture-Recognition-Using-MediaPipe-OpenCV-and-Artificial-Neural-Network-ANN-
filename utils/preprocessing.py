import numpy as np


def normalize_landmarks(landmarks):

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    # 21 landmarks × 3 coordinates
    landmarks = landmarks.reshape(
        21,
        3
    )

    # Use wrist landmark as reference
    reference = landmarks[0].copy()

    # Normalize relative to wrist
    landmarks = landmarks - reference

    return landmarks.flatten()