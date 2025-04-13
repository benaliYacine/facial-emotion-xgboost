import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb
import joblib
import pandas as pd
import math
from pathlib import Path

# Load the saved model and preprocessing components
model = xgb.Booster()
model.load_model("facial_expression_model2.json")
scaler = joblib.load("scaler2.pkl")
label_encoder = joblib.load("label_encoder2.pkl")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Define facial feature extraction functions
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_angle(point1, point2, point3):
    try:
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

        # Normalize vectors
        vector1_norm = np.linalg.norm(vector1)
        vector2_norm = np.linalg.norm(vector2)

        # Check for zero vectors
        if vector1_norm == 0 or vector2_norm == 0:
            return 0.0

        vector1_normalized = vector1 / vector1_norm
        vector2_normalized = vector2 / vector2_norm

        # Calculate dot product and clip to valid range [-1, 1]
        dot_product = np.clip(np.dot(vector1_normalized, vector2_normalized), -1.0, 1.0)

        return math.degrees(math.acos(dot_product))
    except Exception as e:
        return 0.0


def calculate_area(points):
    try:
        if len(points) < 3:
            return 0.0

        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]

        return 0.5 * abs(
            np.sum(x[:-1] * y[1:] + x[-1] * y[0] - x[1:] * y[:-1] - x[0] * y[-1])
        )
    except Exception:
        return 0.0


def extract_features(landmarks, image_width, image_height):
    try:
        points = [
            (int(landmark.x * image_width), int(landmark.y * image_height))
            for landmark in landmarks
        ]

        # Normalize factor (using inter-ocular distance)
        left_eye_center = np.mean([points[33], points[133]], axis=0)
        right_eye_center = np.mean([points[362], points[263]], axis=0)
        normalize_factor = calculate_distance(left_eye_center, right_eye_center)

        if normalize_factor == 0:
            return None

        features = {}

        # A. Distances (normalized)
        features["eyebrow_eye_dist"] = (
            calculate_distance(points[66], points[159]) / normalize_factor
        )
        features["mouth_nose_dist"] = (
            calculate_distance(points[0], points[17]) / normalize_factor
        )
        features["inner_outer_lip_dist"] = (
            calculate_distance(points[13], points[14]) / normalize_factor
        )
        features["eye_width"] = (
            calculate_distance(points[33], points[133]) / normalize_factor
        )
        features["eye_height"] = (
            calculate_distance(points[159], points[145]) / normalize_factor
        )

        # B. Angles
        features["eyebrow_arch"] = calculate_angle(points[66], points[67], points[68])
        features["mouth_corner_angle"] = calculate_angle(
            points[61], points[0], points[291]
        )
        features["nose_bridge_angle"] = calculate_angle(
            points[168], points[6], points[197]
        )

        # C. Ratios
        eye_width = features["eye_width"] * normalize_factor
        if eye_width > 0:
            features["eye_aspect_ratio"] = (
                features["eye_height"] / features["eye_width"]
            )
        else:
            features["eye_aspect_ratio"] = 0.0

        mouth_width = calculate_distance(points[78], points[308])
        if mouth_width > 0:
            features["mouth_aspect_ratio"] = (
                calculate_distance(points[13], points[14]) / mouth_width
            )
        else:
            features["mouth_aspect_ratio"] = 0.0

        # D. Areas (normalized)
        mouth_points = [points[78], points[308], points[13], points[14]]
        left_eye_points = [points[33], points[133], points[159], points[145]]
        features["mouth_area"] = calculate_area(mouth_points) / (normalize_factor**2)
        features["eye_area"] = calculate_area(left_eye_points) / (normalize_factor**2)

        # NEW FEATURES

        # 1. Facial Symmetry Metrics
        features["eye_height_symmetry"] = (
            abs(
                calculate_distance(points[159], points[145])
                - calculate_distance(points[386], points[374])
            )
            / normalize_factor
        )

        features["mouth_corner_symmetry"] = (
            abs(
                calculate_distance(points[61], points[0])
                - calculate_distance(points[291], points[0])
            )
            / normalize_factor
        )

        # 2. Face Width-to-Height Ratio
        face_width = calculate_distance(points[454], points[234])
        face_height = calculate_distance(points[10], points[152])
        features["face_width_height_ratio"] = (
            face_width / face_height if face_height > 0 else 0.0
        )

        # 3. Lower/Upper Face Ratio
        midpoint = (points[10][1] + points[152][1]) / 2
        upper_face_height = midpoint - points[10][1]
        lower_face_height = points[152][1] - midpoint
        features["lower_upper_face_ratio"] = (
            lower_face_height / upper_face_height if upper_face_height > 0 else 0.0
        )

        # 4. Lip Curvature
        top_lip_middle = points[13]
        left_lip_corner = points[61]
        right_lip_corner = points[291]
        lip_corner_midpoint = (
            (left_lip_corner[0] + right_lip_corner[0]) / 2,
            (left_lip_corner[1] + right_lip_corner[1]) / 2,
        )
        features["top_lip_curvature"] = (
            lip_corner_midpoint[1] - top_lip_middle[1]
        ) / normalize_factor

        # 5. Nose Wrinkle Features
        nose_top = points[6]
        nose_bottom = points[4]
        features["nose_wrinkle"] = (
            calculate_distance(nose_top, nose_bottom) / normalize_factor
        )

        # 6. Brow Features
        left_inner_brow = points[65]
        left_outer_brow = points[105]
        right_inner_brow = points[295]
        right_outer_brow = points[334]
        features["brow_inner_distance"] = (
            calculate_distance(left_inner_brow, right_inner_brow) / normalize_factor
        )

        # Safely calculate brow slopes with zero division protection
        left_dx = left_outer_brow[0] - left_inner_brow[0]
        right_dx = right_outer_brow[0] - right_inner_brow[0]

        features["left_brow_slope"] = (
            (left_outer_brow[1] - left_inner_brow[1]) / left_dx if left_dx != 0 else 0
        )
        features["right_brow_slope"] = (
            (right_outer_brow[1] - right_inner_brow[1]) / right_dx
            if right_dx != 0
            else 0
        )

        # 7. Cheek Features
        left_cheek = points[117]
        right_cheek = points[346]
        features["cheek_distance"] = (
            calculate_distance(left_cheek, right_cheek) / normalize_factor
        )

        # 8. Mouth Openness Refined
        upper_lip_top = points[13]
        lower_lip_bottom = points[14]
        features["vertical_mouth_openness"] = (
            calculate_distance(upper_lip_top, lower_lip_bottom) / normalize_factor
        )

        # 9. Advanced Eye Features
        left_eye_top = points[159]
        left_eye_bottom = points[145]
        features["left_eye_openness"] = (
            calculate_distance(left_eye_top, left_eye_bottom) / normalize_factor
        )
        features["right_eye_openness"] = (
            calculate_distance(points[386], points[374]) / normalize_factor
        )

        # Avoid division by zero for ratio
        if features["right_eye_openness"] > 0:
            features["eye_openness_ratio"] = (
                features["left_eye_openness"] / features["right_eye_openness"]
            )
        else:
            features["eye_openness_ratio"] = 1.0

        # 10. Dynamic Wrinkle Approximations
        forehead_point = points[10]
        brow_mid_point = points[67]
        features["forehead_height"] = (
            forehead_point[1] - brow_mid_point[1]
        ) / normalize_factor

        return features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def predict_emotion(features, scaler, model, label_encoder):
    try:
        # Convert features to a DataFrame to ensure consistent order
        feature_df = pd.DataFrame([features])

        # Make sure feature order matches what the model expects
        expected_features = [
            "eyebrow_eye_dist",
            "mouth_nose_dist",
            "inner_outer_lip_dist",
            "eye_width",
            "eye_height",
            "eyebrow_arch",
            "mouth_corner_angle",
            "nose_bridge_angle",
            "eye_aspect_ratio",
            "mouth_aspect_ratio",
            "mouth_area",
            "eye_area",
            # Add new features here if your model has been trained with them
            # If using with a model that wasn't trained with these features,
            # comment these out or retrain your model
            "eye_height_symmetry",
            "mouth_corner_symmetry",
            "face_width_height_ratio",
            "lower_upper_face_ratio",
            "top_lip_curvature",
            "nose_wrinkle",
            "brow_inner_distance",
            "left_brow_slope",
            "right_brow_slope",
            "cheek_distance",
            "vertical_mouth_openness",
            "left_eye_openness",
            "right_eye_openness",
            "eye_openness_ratio",
            "forehead_height",
        ]

        # Filter to only include features that exist in the input
        available_features = [f for f in expected_features if f in feature_df.columns]

        # If using with existing model trained on original features only:
        # available_features = expected_features[:12]  # Use only the first 12 original features

        # Reorder columns to match training data
        feature_df = feature_df[available_features]
        features_array = feature_df.values

        # Scale features
        features_scaled = scaler.transform(features_array)

        # Create DMatrix for XGBoost prediction
        dfeatures = xgb.DMatrix(features_scaled)

        # Make prediction
        prediction = model.predict(dfeatures)
        emotion = label_encoder.inverse_transform([int(prediction[0])])[0]
        return emotion
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Unknown"


# Set up the webcam
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Press 'q' to quit")

emotion_colors = {
    "Angry": (0, 0, 255),  # Red
    "Happy": (0, 255, 255),  # Yellow
    "Neutral": (255, 255, 255),  # White
    "Sad": (255, 0, 0),  # Blue
    "Surprise": (0, 255, 0),  # Green
}


prediction_history = []
history_size = 5

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame from camera. Exiting...")
        break

    # Flip the frame horizontally for a more natural view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # Process the frame and detect faces
    results = face_mesh.process(rgb_frame)

    # Count detected faces
    num_faces = 0
    detected_emotion = "No Face"

    if results.multi_face_landmarks:
        num_faces = len(results.multi_face_landmarks)

        # Process the first detected face
        face_landmarks = results.multi_face_landmarks[0]

        # Extract facial features
        features = extract_features(face_landmarks.landmark, w, h)

        if features:
            # Predict emotion from features
            emotion = predict_emotion(features, scaler, model, label_encoder)

            # Add to history for smoothing
            prediction_history.append(emotion)
            if len(prediction_history) > history_size:
                prediction_history.pop(0)

            # Get most common prediction from history
            if prediction_history:
                emotion_counts = {}
                for e in prediction_history:
                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
                detected_emotion = max(emotion_counts, key=emotion_counts.get)
            else:
                detected_emotion = emotion

            # Get color for detected emotion
            emotion_color = emotion_colors.get(detected_emotion, (255, 255, 255))

            # Draw the face mesh (using default style, not custom color)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            # Draw the face contours
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

            # Draw a colored rectangle based on emotion at the top of the frame
            cv2.rectangle(frame, (0, 0), (w, 40), emotion_color, -1)

            # Display the predicted emotion
            cv2.putText(
                frame,
                f"Emotion: {detected_emotion}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),  # Black text on colored background
                2,
            )
        else:
            # If features couldn't be extracted
            cv2.putText(
                frame,
                "Cannot extract features",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
    else:
        # Reset prediction history when no face is detected
        prediction_history = []

        # Display "No face detected" message
        cv2.putText(
            frame,
            "No face detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    # Display the frame with detections
    cv2.imshow("Facial Expression Recognition", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
