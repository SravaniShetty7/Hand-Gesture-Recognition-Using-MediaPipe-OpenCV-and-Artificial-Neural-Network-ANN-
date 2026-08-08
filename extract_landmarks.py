
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "leapGestRecog"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker (1).task"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "landmark_dataset.csv"
)

# ============================================================
# CHECK PATHS
# ============================================================

print("Dataset path:")
print(DATASET_PATH)

print("\nModel path:")
print(MODEL_PATH)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"MediaPipe model not found:\n{MODEL_PATH}"
    )

# ============================================================
# GESTURE MAPPING
# ============================================================

gesture_names = {
    "00": "Palm",
    "01": "L",
    "02": "Fist",
    "03": "Fist_Moved",
    "04": "Thumb",
    "05": "Index",
    "06": "OK",
    "07": "Palm_Moved",
    "08": "C",
    "09": "Down"
}

# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# ============================================================
# STORAGE
# ============================================================

data = []

total_images = 0
successful = 0
failed = 0

# ============================================================
# PROCESS DATASET
# ============================================================

# Only process folders 00 to 09
gesture_folders = [
    folder for folder in sorted(os.listdir(DATASET_PATH))
    if folder in gesture_names
]

print("\nGesture folders found:")
print(gesture_folders)

# ============================================================
# LOOP THROUGH GESTURES
# ============================================================

for gesture_folder in gesture_folders:

    gesture_path = os.path.join(
        DATASET_PATH,
        gesture_folder
    )

    label = gesture_names[gesture_folder]

    print("\n======================================")
    print(f"Processing: {gesture_folder} -> {label}")
    print("======================================")

    if not os.path.isdir(gesture_path):
        continue

    # --------------------------------------------------------
    # PERSON FOLDERS
    # Example:
    # 00/
    #    01_palm/
    #    02_palm/
    #    ...
    # --------------------------------------------------------

    person_folders = sorted(
        os.listdir(gesture_path)
    )

    for person_folder in person_folders:

        person_path = os.path.join(
            gesture_path,
            person_folder
        )

        if not os.path.isdir(person_path):
            continue

        print(
            f"  Person folder: {person_folder}"
        )

        image_files = sorted(
            os.listdir(person_path)
        )

        # ----------------------------------------------------
        # PROCESS EACH IMAGE
        # ----------------------------------------------------

        for image_name in image_files:

            image_path = os.path.join(
                person_path,
                image_name
            )

            if not image_name.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                continue

            total_images += 1

            image = cv2.imread(image_path)

            if image is None:
                failed += 1
                continue

            # ------------------------------------------------
            # DO NOT RESIZE
            # This keeps preprocessing consistent.
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # ------------------------------------------------
            # DETECT HAND
            # ------------------------------------------------

            result = landmarker.detect(
                mp_image
            )

            if not result.hand_landmarks:
                failed += 1
                continue

            landmarks = result.hand_landmarks[0]

            if len(landmarks) != 21:
                failed += 1
                continue

            # ------------------------------------------------
            # EXTRACT 21 x 3 = 63 FEATURES
            # ------------------------------------------------

            row = []

            for landmark in landmarks:

                row.append(
                    float(landmark.x)
                )

                row.append(
                    float(landmark.y)
                )

                row.append(
                    float(landmark.z)
                )

            # Safety check
            if len(row) != 63:
                failed += 1
                continue

            # ------------------------------------------------
            # ADD LABEL
            # ------------------------------------------------

            row.append(label)

            data.append(row)

            successful += 1

# ============================================================
# CLOSE MEDIAPIPE
# ============================================================

landmarker.close()

# ============================================================
# CREATE DATAFRAME
# ============================================================

feature_columns = []

for i in range(21):

    feature_columns.append(f"x{i}")
    feature_columns.append(f"y{i}")
    feature_columns.append(f"z{i}")

columns = feature_columns + ["label"]

df = pd.DataFrame(
    data,
    columns=columns
)

# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("LANDMARK EXTRACTION COMPLETED")
print("======================================")

print(
    f"Total images checked : {total_images}"
)

print(
    f"Successful detections: {successful}"
)

print(
    f"Failed detections    : {failed}"
)

print(
    f"CSV rows             : {len(df)}"
)

print(
    f"CSV columns          : {len(df.columns)}"
)

print("\nClass distribution:")

print(
    df["label"].value_counts()
)

print(
    f"\nCSV saved at:\n{OUTPUT_CSV}"
)

print("\nFirst 5 rows:")

print(
    df.head()
)

