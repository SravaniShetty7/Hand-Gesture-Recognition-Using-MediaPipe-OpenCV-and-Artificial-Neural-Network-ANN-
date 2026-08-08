import os
import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# MODEL FILES
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

print("Checking model files...")

print("Weights:", WEIGHTS_PATH)
print("Scaler :", SCALER_PATH)
print("Encoder:", ENCODER_PATH)


if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"\nANN weights not found:\n{WEIGHTS_PATH}"
    )

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"\nScaler not found:\n{SCALER_PATH}"
    )

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        f"\nLabel encoder not found:\n{ENCODER_PATH}"
    )


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(
    SCALER_PATH
)

print(
    "Scaler features:",
    scaler.n_features_in_
)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

label_encoder = joblib.load(
    ENCODER_PATH
)

print(
    "Classes:",
    label_encoder.classes_
)


# ============================================================
# RECREATE ANN
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
# LOAD WEIGHTS
# ============================================================

model.load_weights(
    WEIGHTS_PATH
)


print("ANN model loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_gesture(features):

    if features is None:
        return None, 0.0


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    features = np.asarray(
        features,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Check 63 features
    # --------------------------------------------------------

    if features.size != 63:

        raise ValueError(
            f"Expected 63 features, "
            f"but received {features.size}"
        )


    # --------------------------------------------------------
    # Reshape
    # --------------------------------------------------------

    features = features.reshape(
        1,
        63
    )


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    features_scaled = scaler.transform(
        features
    )


    # --------------------------------------------------------
    # ANN PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict(
        features_scaled,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # BEST CLASS
    # --------------------------------------------------------

    predicted_index = np.argmax(
        probabilities
    )


    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    gesture = label_encoder.inverse_transform(
        [predicted_index]
    )[0]


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = float(
        probabilities[predicted_index]
    )


    return gesture, confidence