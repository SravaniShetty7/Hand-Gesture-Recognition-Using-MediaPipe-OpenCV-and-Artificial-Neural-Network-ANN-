import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "new"
)


# ============================================================
# FILE PATHS
# ============================================================

WEIGHTS_PATH = os.path.join(
    MODEL_DIR,
    "new_hand_gesture_weights.weights.h5"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "new_scaler.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "new_label_encoder.pkl"
)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    WEIGHTS_PATH,
    SCALER_PATH,
    ENCODER_PATH
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(SCALER_PATH)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

label_encoder = joblib.load(ENCODER_PATH)

print("Scaler features:", scaler.n_features_in_)
print("Classes:", label_encoder.classes_)


# ============================================================
# BUILD ANN
# ============================================================

model = Sequential([
    Input(shape=(63,)),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# BUILD MODEL
# ============================================================

model.build(
    input_shape=(None, 63)
)


# ============================================================
# LOAD ONLY WEIGHTS
# ============================================================

model.load_weights(
    WEIGHTS_PATH
)

print("ANN model loaded successfully!")


# ============================================================
# PREDICT GESTURE
# ============================================================

def predict_gesture(landmarks):

    if landmarks is None:
        return None, 0.0

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # CHECK FEATURES
    # --------------------------------------------------------

    if landmarks.shape != (63,):

        landmarks = landmarks.reshape(-1)

        if len(landmarks) != 63:
            return None, 0.0


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    landmarks_scaled = scaler.transform(
        landmarks.reshape(1, -1)
    )


    # --------------------------------------------------------
    # ANN PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict(
        landmarks_scaled,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # BEST CLASS
    # --------------------------------------------------------

    prediction_index = np.argmax(
        probabilities
    )

    confidence = float(
        probabilities[prediction_index]
    )


    # --------------------------------------------------------
    # DECODE LABEL
    # --------------------------------------------------------

    gesture = label_encoder.inverse_transform(
        [prediction_index]
    )[0]


    return gesture, confidence