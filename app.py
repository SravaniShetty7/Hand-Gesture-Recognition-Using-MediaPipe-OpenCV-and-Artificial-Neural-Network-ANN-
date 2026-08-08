import os
import cv2
import numpy as np
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)

from predictor import predict_gesture

from utils.mediapipe_helper import (
    extract_landmarks_image,
    extract_landmarks_video
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hand Gesture AI",
    page_icon="✋",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("✋ Hand Gesture AI")

st.write(
    "ANN + MediaPipe + OpenCV"
)

st.write(
    "Recognize hand gestures using 21 MediaPipe landmarks "
    "and an Artificial Neural Network."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Select Input Method"
)


option = st.sidebar.radio(
    "Choose one:",
    [
        "📤 Upload Image",
        "📷 Take Picture",
        "🎥 Live Webcam"
    ]
)


# ============================================================
# GESTURE DISPLAY
# ============================================================

def show_prediction(
    gesture,
    confidence
):

    if gesture is None:

        st.warning(
            "No hand detected."
        )

        return


    st.success(
        f"Gesture: {gesture}"
    )


    st.info(
        f"Confidence: {confidence * 100:.2f}%"
    )


    # Progress bar

    st.progress(
        min(
            max(confidence, 0.0),
            1.0
        )
    )


# ============================================================
# DRAW PREDICTION ON FRAME
# ============================================================

def draw_prediction(
    frame,
    gesture,
    confidence
):

    if gesture is None:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return frame


    text = (
        f"{gesture} "
        f"{confidence * 100:.1f}%"
    )


    cv2.rectangle(
        frame,
        (10, 10),
        (390, 75),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        text,
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3
    )


    return frame


# ============================================================
# OPTION 1
# UPLOAD IMAGE
# ============================================================

if option == "📤 Upload Image":

    st.header(
        "📤 Upload Image"
    )


    uploaded_file = st.file_uploader(
        "Choose a hand image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )


        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if image is None:

            st.error(
                "Unable to read image."
            )

        else:

            # Display image

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Uploaded Image",
                use_container_width=True
            )


            # Extract landmarks

            landmarks = (
                extract_landmarks_image(
                    image
                )
            )


            if landmarks is None:

                st.warning(
                    "No hand detected in the image."
                )

            else:

                gesture, confidence = (
                    predict_gesture(
                        landmarks
                    )
                )


                show_prediction(
                    gesture,
                    confidence
                )


# ============================================================
# OPTION 2
# TAKE PICTURE
# ============================================================

elif option == "📷 Take Picture":

    st.header(
        "📷 Take Picture"
    )


    camera_image = st.camera_input(
        "Take a picture of your hand"
    )


    if camera_image is not None:

        file_bytes = np.asarray(
            bytearray(
                camera_image.getvalue()
            ),
            dtype=np.uint8
        )


        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if image is None:

            st.error(
                "Unable to read camera image."
            )

        else:

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Captured Image",
                use_container_width=True
            )


            # Extract MediaPipe landmarks

            landmarks = (
                extract_landmarks_image(
                    image
                )
            )


            if landmarks is None:

                st.warning(
                    "No hand detected."
                )

            else:

                gesture, confidence = (
                    predict_gesture(
                        landmarks
                    )
                )


                show_prediction(
                    gesture,
                    confidence
                )


# ============================================================
# OPTION 3
# LIVE WEBCAM
# ============================================================

elif option == "🎥 Live Webcam":

    st.header(
        "🎥 Live Webcam"
    )


    st.write(
        "Click START below and allow camera access."
    )


    # ========================================================
    # VIDEO PROCESSOR
    # ========================================================

    class GestureVideoProcessor(
        VideoProcessorBase
    ):

        def __init__(self):

            self.gesture = None

            self.confidence = 0.0


        def recv(
            self,
            frame
        ):

            # Convert WebRTC frame to BGR

            img = frame.to_ndarray(
                format="bgr24"
            )


            # Mirror webcam

            img = cv2.flip(
                img,
                1
            )


            # Extract landmarks

            landmarks = (
                extract_landmarks_video(
                    img
                )
            )


            if landmarks is None:

                self.gesture = None

                self.confidence = 0.0


            else:

                try:

                    gesture, confidence = (
                        predict_gesture(
                            landmarks
                        )
                    )


                    self.gesture = gesture

                    self.confidence = confidence


                except Exception:

                    self.gesture = None

                    self.confidence = 0.0


            # Draw result

            img = draw_prediction(
                img,
                self.gesture,
                self.confidence
            )


            # Convert back

            return frame.from_ndarray(
                img,
                format="bgr24"
            )


    # ========================================================
    # WEBRTC
    # ========================================================

    webrtc_ctx = webrtc_streamer(

        key="hand-gesture-webcam",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=(
            GestureVideoProcessor
        ),

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )


    st.write("")


    st.info(
        "Show one hand clearly inside the camera. "
        "Keep the hand reasonably close to the camera."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Hand Gesture Recognition | "
    "MediaPipe Hand Landmarker + ANN + Streamlit"
)