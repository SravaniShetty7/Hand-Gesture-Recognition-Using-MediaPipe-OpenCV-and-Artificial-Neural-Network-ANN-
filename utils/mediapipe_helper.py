import os
import time
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# MEDIAPIPE MODEL
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "new",
    "hand_landmarker (1).task"
)


# ============================================================
# IMPORTANT:
# If your .task file is directly inside models,
# use this instead:
#
# MODEL_PATH = os.path.join(
#     BASE_DIR,
#     "models",
#     "hand_landmarker (1).task"
# )
# ============================================================


if not os.path.exists(MODEL_PATH):

    # Try alternative location
    MODEL_PATH = os.path.join(
        BASE_DIR,
        "models",
        "hand_landmarker (1).task"
    )


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nMediaPipe model not found.\n"
        f"Expected location:\n{MODEL_PATH}"
    )


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = python.BaseOptions

HandLandmarker = vision.HandLandmarker

HandLandmarkerOptions = vision.HandLandmarkerOptions

RunningMode = vision.RunningMode


# ============================================================
# IMAGE LANDMARKER
# ============================================================

image_options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.IMAGE,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


image_landmarker = (
    HandLandmarker.create_from_options(
        image_options
    )
)


# ============================================================
# VIDEO LANDMARKER
# ============================================================

video_options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


video_landmarker = (
    HandLandmarker.create_from_options(
        video_options
    )
)


# ============================================================
# VIDEO TIMESTAMP
# ============================================================

START_TIME = time.perf_counter()


# ============================================================
# LANDMARKS → 63 FEATURES
# ============================================================

def landmarks_to_features(result):

    if result is None:
        return None


    if not result.hand_landmarks:
        return None


    landmarks = result.hand_landmarks[0]


    if len(landmarks) != 21:
        return None


    features = []


    for point in landmarks:

        features.append(
            point.x
        )

        features.append(
            point.y
        )

        features.append(
            point.z
        )


    features = np.asarray(
        features,
        dtype=np.float32
    )


    if features.shape != (63,):
        return None


    return features


# ============================================================
# IMAGE LANDMARK EXTRACTION
# ============================================================

def extract_landmarks_image(image):

    if image is None:
        return None


    # BGR → RGB

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    result = image_landmarker.detect(
        mp_image
    )


    return landmarks_to_features(
        result
    )


# ============================================================
# VIDEO LANDMARK EXTRACTION
# ============================================================

def extract_landmarks_video(frame):

    if frame is None:
        return None


    # BGR → RGB

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp = int(
        (
            time.perf_counter()
            - START_TIME
        ) * 1000
    )


    if timestamp <= 0:
        timestamp = 1


    result = (
        video_landmarker.detect_for_video(
            mp_image,
            timestamp
        )
    )


    return landmarks_to_features(
        result
    )