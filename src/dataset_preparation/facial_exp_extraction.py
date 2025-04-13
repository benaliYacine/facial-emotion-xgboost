import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math
import random


class FacialFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_angle(self, point1, point2, point3):
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
            dot_product = np.clip(
                np.dot(vector1_normalized, vector2_normalized), -1.0, 1.0
            )

            return math.degrees(math.acos(dot_product))
        except Exception as e:
            print(f"Warning: Error calculating angle: {e}")
            return 0.0

    def calculate_area(self, points):
        try:
            if len(points) < 3:
                return 0.0

            points = np.array(points)
            x = points[:, 0]
            y = points[:, 1]

            return 0.5 * abs(
                np.sum(x[:-1] * y[1:] + x[-1] * y[0] - x[1:] * y[:-1] - x[0] * y[-1])
            )
        except Exception as e:
            print(f"Warning: Error calculating area: {e}")
            return 0.0

    def extract_features(self, image_path):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

            results = self.face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                print(f"Warning: No face detected in {image_path}")
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            points = [
                (int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks
            ]

            # Normalize factor (using inter-ocular distance)
            left_eye_center = np.mean([points[33], points[133]], axis=0)
            right_eye_center = np.mean([points[362], points[263]], axis=0)
            normalize_factor = self.calculate_distance(
                left_eye_center, right_eye_center
            )

            if normalize_factor == 0:
                print(f"Warning: Invalid normalization factor in {image_path}")
                return None

            features = {}

            # A. Distances (normalized)
            features["eyebrow_eye_dist"] = (
                self.calculate_distance(points[66], points[159]) / normalize_factor
            )
            features["mouth_nose_dist"] = (
                self.calculate_distance(points[0], points[17]) / normalize_factor
            )
            features["inner_outer_lip_dist"] = (
                self.calculate_distance(points[13], points[14]) / normalize_factor
            )
            features["eye_width"] = (
                self.calculate_distance(points[33], points[133]) / normalize_factor
            )
            features["eye_height"] = (
                self.calculate_distance(points[159], points[145]) / normalize_factor
            )

            # B. Angles
            features["eyebrow_arch"] = self.calculate_angle(
                points[66], points[67], points[68]
            )
            features["mouth_corner_angle"] = self.calculate_angle(
                points[61], points[0], points[291]
            )
            features["nose_bridge_angle"] = self.calculate_angle(
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

            mouth_width = self.calculate_distance(points[78], points[308])
            if mouth_width > 0:
                features["mouth_aspect_ratio"] = (
                    self.calculate_distance(points[13], points[14]) / mouth_width
                )
            else:
                features["mouth_aspect_ratio"] = 0.0

            # D. Areas (normalized)
            mouth_points = [points[78], points[308], points[13], points[14]]
            left_eye_points = [points[33], points[133], points[159], points[145]]
            features["mouth_area"] = self.calculate_area(mouth_points) / (
                normalize_factor**2
            )
            features["eye_area"] = self.calculate_area(left_eye_points) / (
                normalize_factor**2
            )

            # 1. Facial Symmetry Metrics
            features["eye_height_symmetry"] = (
                abs(
                    self.calculate_distance(points[159], points[145])
                    - self.calculate_distance(points[386], points[374])
                )
                / normalize_factor
            )
            features["mouth_corner_symmetry"] = (
                abs(
                    self.calculate_distance(points[61], points[0])
                    - self.calculate_distance(points[291], points[0])
                )
                / normalize_factor
            )

            # 2. Face Width-to-Height Ratio
            face_width = self.calculate_distance(points[454], points[234])
            face_height = self.calculate_distance(points[10], points[152])
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

            # 4. Lip Curvature (important for distinguishing smiles from neutral)
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

            # 5. Nose Wrinkle Features (important for disgust)
            nose_top = points[6]
            nose_bottom = points[4]
            features["nose_wrinkle"] = (
                self.calculate_distance(nose_top, nose_bottom) / normalize_factor
            )

            # 6. Brow Features (critical for anger, surprise, sadness)
            left_inner_brow = points[65]
            left_outer_brow = points[105]
            right_inner_brow = points[295]
            right_outer_brow = points[334]
            features["brow_inner_distance"] = (
                self.calculate_distance(left_inner_brow, right_inner_brow)
                / normalize_factor
            )
            features["left_brow_slope"] = (
                (left_outer_brow[1] - left_inner_brow[1])
                / (left_outer_brow[0] - left_inner_brow[0])
                if (left_outer_brow[0] - left_inner_brow[0]) != 0
                else 0
            )
            features["right_brow_slope"] = (
                (right_outer_brow[1] - right_inner_brow[1])
                / (right_outer_brow[0] - right_inner_brow[0])
                if (right_outer_brow[0] - right_inner_brow[0]) != 0
                else 0
            )

            # 7. Cheek Features
            left_cheek = points[117]
            right_cheek = points[346]
            features["cheek_distance"] = (
                self.calculate_distance(left_cheek, right_cheek) / normalize_factor
            )

            # 8. Mouth Openness Refined
            upper_lip_top = points[13]
            lower_lip_bottom = points[14]
            features["vertical_mouth_openness"] = (
                self.calculate_distance(upper_lip_top, lower_lip_bottom)
                / normalize_factor
            )

            # 9. Advanced Eye Features
            left_eye_top = points[159]
            left_eye_bottom = points[145]
            left_eye_inner = points[133]
            left_eye_outer = points[33]
            features["left_eye_openness"] = (
                self.calculate_distance(left_eye_top, left_eye_bottom)
                / normalize_factor
            )
            features["right_eye_openness"] = (
                self.calculate_distance(points[386], points[374]) / normalize_factor
            )
            features["eye_openness_ratio"] = (
                features["left_eye_openness"] / features["right_eye_openness"]
                if features["right_eye_openness"] > 0
                else 1.0
            )

            # 10. Dynamic Wrinkle Approximations (important for certain emotional expressions)
            # Forehead area compared to brow height
            forehead_point = points[10]
            brow_mid_point = points[67]
            features["forehead_height"] = (
                forehead_point[1] - brow_mid_point[1]
            ) / normalize_factor

            return features

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None


def process_dataset(base_path):
    extractor = FacialFeatureExtractor()
    data = []

    base_path = Path(base_path)
    total_images = 0
    processed_images = 0

    for emotion_dir in base_path.glob("*"):
        if emotion_dir.is_dir():
            emotion = emotion_dir.name

            # Skip the Ahegao folder
            if emotion.lower() == "ahegao":
                print(f"Skipping folder: {emotion}")
                continue

            for img_path in emotion_dir.glob("*.[pP][nN][gG]"):
                total_images += 1
                features = extractor.extract_features(img_path)
                if features:
                    features["emotion"] = emotion
                    features["image_path"] = str(img_path)
                    data.append(features)
                    processed_images += 1

    print(f"\nProcessing Summary:")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Failed to process: {total_images - processed_images}")

    if not data:
        raise ValueError("No features could be extracted from any images")

    df = pd.DataFrame(data)
    return df


def visualize_example_images(base_path, num_examples=1):
    """
    Visualize example images from each emotion category with face mesh overlay

    Args:
        base_path: Path to the dataset directory
        num_examples: Number of examples to show per emotion
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    base_path = Path(base_path)

    # Create a face mesh instance for visualization
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:

        for emotion_dir in base_path.glob("*"):
            if emotion_dir.is_dir():
                # Skip the Ahegao folder
                if emotion_dir.name.lower() == "ahegao":
                    print(f"Skipping folder: {emotion_dir.name}")
                    continue

                emotion = emotion_dir.name
                print(f"\nEmotion: {emotion}")

                # Get list of image files
                image_files = list(emotion_dir.glob("*.[pP][nN][gG]"))
                if not image_files:
                    print(f"No images found for {emotion}")
                    continue

                # Select random examples
                examples = random.sample(
                    image_files, min(num_examples, len(image_files))
                )

                for img_path in examples:
                    # Read the image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"Could not read image: {img_path}")
                        continue

                    # Convert to RGB for processing
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb)

                    # Draw face mesh
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                            )
                    else:
                        print(f"No face detected in {img_path}")
                        continue

                    # Display the image
                    image = cv2.resize(
                        image, (0, 0), fx=0.5, fy=0.5
                    )  # Resize for better display
                    cv2.imshow(f"{emotion} - {img_path.name}", image)
                    cv2.waitKey(0)

        cv2.destroyAllWindows()


def visualize_all_facial_features(image_path, output_folder="facial_visualizations"):
    """
    Visualize all facial features being calculated and save visualizations to a folder.

    Args:
        image_path: Path to the image to visualize
        output_folder: Folder to save visualizations
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize facial feature extractor
    extractor = FacialFeatureExtractor()

    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # Process image with face mesh
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    results = extractor.face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        print(f"No face detected in {image_path}")
        return

    landmarks = results.multi_face_landmarks[0].landmark
    points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks]

    # Create a base image with face mesh
    base_image = image.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=base_image,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

    # Save base mesh image
    cv2.imwrite(str(output_dir / "base_mesh.jpg"), base_image)

    # Dictionary to store visualizations
    visualizations = {}

    # Normalization - inter-ocular distance
    vis = image.copy()
    left_eye_center = np.mean([points[33], points[133]], axis=0).astype(int)
    right_eye_center = np.mean([points[362], points[263]], axis=0).astype(int)
    cv2.circle(vis, tuple(left_eye_center), 5, (0, 255, 0), -1)
    cv2.circle(vis, tuple(right_eye_center), 5, (0, 255, 0), -1)
    cv2.line(vis, tuple(left_eye_center), tuple(right_eye_center), (0, 255, 0), 2)
    cv2.putText(
        vis,
        "Normalization (inter-ocular distance)",
        (
            int((left_eye_center[0] + right_eye_center[0]) / 2) - 100,
            left_eye_center[1] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    visualizations["01_normalization"] = vis

    # A. Distances section
    # 1. Eyebrow to Eye distance
    vis = image.copy()
    cv2.line(vis, points[66], points[159], (255, 0, 0), 2)
    cv2.putText(
        vis,
        "eyebrow_eye_dist",
        (points[66][0], points[66][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
    )
    visualizations["02_eyebrow_eye_dist"] = vis

    # 2. Mouth to Nose distance
    vis = image.copy()
    cv2.line(vis, points[0], points[17], (0, 0, 255), 2)
    cv2.putText(
        vis,
        "mouth_nose_dist",
        (points[0][0] - 40, (points[0][1] + points[17][1]) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    visualizations["03_mouth_nose_dist"] = vis

    # 3. Inner to Outer lip distance
    vis = image.copy()
    cv2.line(vis, points[13], points[14], (255, 255, 0), 2)
    cv2.putText(
        vis,
        "inner_outer_lip_dist",
        (points[13][0] - 70, points[13][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )
    visualizations["04_inner_outer_lip_dist"] = vis

    # 4. Eye width
    vis = image.copy()
    cv2.line(vis, points[33], points[133], (0, 255, 255), 2)
    cv2.putText(
        vis,
        "eye_width",
        (points[33][0], points[33][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    visualizations["05_eye_width"] = vis

    # 5. Eye height
    vis = image.copy()
    cv2.line(vis, points[159], points[145], (0, 255, 255), 2)
    cv2.putText(
        vis,
        "eye_height",
        (points[159][0] + 5, points[159][1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    visualizations["06_eye_height"] = vis

    # B. Angles section
    # 1. Eyebrow arch
    vis = image.copy()
    p1, p2, p3 = points[66], points[67], points[68]
    cv2.line(vis, p1, p2, (128, 0, 128), 2)
    cv2.line(vis, p2, p3, (128, 0, 128), 2)
    cv2.putText(
        vis,
        "eyebrow_arch",
        (p2[0] - 30, p2[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (128, 0, 128),
        1,
    )
    visualizations["07_eyebrow_arch"] = vis

    # 2. Mouth corner angle
    vis = image.copy()
    p1, p2, p3 = points[61], points[0], points[291]
    cv2.line(vis, p1, p2, (255, 128, 0), 2)
    cv2.line(vis, p2, p3, (255, 128, 0), 2)
    cv2.putText(
        vis,
        "mouth_corner_angle",
        (p2[0] - 60, p2[1] + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 128, 0),
        1,
    )
    visualizations["08_mouth_corner_angle"] = vis

    # 3. Nose bridge angle
    vis = image.copy()
    p1, p2, p3 = points[168], points[6], points[197]
    cv2.line(vis, p1, p2, (0, 128, 255), 2)
    cv2.line(vis, p2, p3, (0, 128, 255), 2)
    cv2.putText(
        vis,
        "nose_bridge_angle",
        (p2[0] - 60, p2[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 128, 255),
        1,
    )
    visualizations["09_nose_bridge_angle"] = vis

    # C. Ratios section
    # 1. Eye aspect ratio visualization
    vis = image.copy()
    cv2.line(vis, points[33], points[133], (0, 255, 255), 2)
    cv2.line(vis, points[159], points[145], (0, 255, 255), 2)
    cv2.putText(
        vis,
        "eye_aspect_ratio = eye_height/eye_width",
        (points[33][0], points[33][1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    visualizations["10_eye_aspect_ratio"] = vis

    # 2. Mouth aspect ratio
    vis = image.copy()
    cv2.line(vis, points[78], points[308], (0, 165, 255), 2)  # mouth width
    cv2.line(vis, points[13], points[14], (0, 165, 255), 2)  # mouth height
    cv2.putText(
        vis,
        "mouth_aspect_ratio = height/width",
        (points[78][0], points[78][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 165, 255),
        1,
    )
    visualizations["11_mouth_aspect_ratio"] = vis

    # D. Areas section
    # 1. Mouth area
    vis = image.copy()
    mouth_points = np.array([points[78], points[308], points[13], points[14]], np.int32)
    cv2.polylines(vis, [mouth_points.reshape((-1, 1, 2))], True, (0, 165, 255), 2)
    cv2.putText(
        vis,
        "mouth_area",
        (int(np.mean(mouth_points[:, 0])) - 40, int(np.mean(mouth_points[:, 1]))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 165, 255),
        1,
    )
    visualizations["12_mouth_area"] = vis

    # 2. Eye area
    vis = image.copy()
    eye_points = np.array([points[33], points[133], points[159], points[145]], np.int32)
    cv2.polylines(vis, [eye_points.reshape((-1, 1, 2))], True, (255, 0, 127), 2)
    cv2.putText(
        vis,
        "eye_area",
        (int(np.mean(eye_points[:, 0])) - 30, int(np.mean(eye_points[:, 1]))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 127),
        1,
    )
    visualizations["13_eye_area"] = vis

    # 1. Facial Symmetry Metrics
    # Eye height symmetry
    vis = image.copy()
    cv2.line(vis, points[159], points[145], (255, 0, 127), 2)  # Left eye height
    cv2.line(vis, points[386], points[374], (255, 0, 127), 2)  # Right eye height
    cv2.putText(
        vis,
        "eye_height_symmetry",
        (
            (points[159][0] + points[386][0]) // 2,
            (points[159][1] + points[386][1]) // 2 - 20,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 127),
        1,
    )
    visualizations["14_eye_height_symmetry"] = vis

    # Mouth corner symmetry
    vis = image.copy()
    cv2.line(vis, points[61], points[0], (255, 128, 0), 2)  # Left corner to center
    cv2.line(vis, points[291], points[0], (255, 128, 0), 2)  # Right corner to center
    cv2.putText(
        vis,
        "mouth_corner_symmetry",
        (points[0][0] - 70, points[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 128, 0),
        1,
    )
    visualizations["15_mouth_corner_symmetry"] = vis

    # 2. Face Width-to-Height Ratio
    vis = image.copy()
    cv2.line(vis, points[454], points[234], (255, 0, 255), 2)  # Face width
    cv2.line(vis, points[10], points[152], (255, 0, 255), 2)  # Face height
    cv2.putText(
        vis,
        "face_width_height_ratio",
        (
            (points[454][0] + points[234][0]) // 2 - 70,
            (points[454][1] + points[234][1]) // 2,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        1,
    )
    visualizations["16_face_width_height_ratio"] = vis

    # 3. Lower/Upper Face Ratio
    vis = image.copy()
    midpoint_y = (points[10][1] + points[152][1]) / 2
    midpoint = (int((points[10][0] + points[152][0]) / 2), int(midpoint_y))
    cv2.line(vis, points[10], midpoint, (128, 128, 0), 2)  # Upper face
    cv2.line(vis, midpoint, points[152], (128, 128, 0), 2)  # Lower face
    cv2.putText(
        vis,
        "upper_face",
        (points[10][0] - 60, (points[10][1] + midpoint[1]) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (128, 128, 0),
        1,
    )
    cv2.putText(
        vis,
        "lower_face",
        (midpoint[0] - 60, (midpoint[1] + points[152][1]) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (128, 128, 0),
        1,
    )
    cv2.putText(
        vis,
        "lower_upper_face_ratio",
        (midpoint[0] - 70, midpoint[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (128, 128, 0),
        1,
    )
    visualizations["17_lower_upper_face_ratio"] = vis

    # 4. Lip Curvature
    vis = image.copy()
    top_lip_middle = points[13]
    left_lip_corner = points[61]
    right_lip_corner = points[291]
    lip_corner_midpoint = (
        int((left_lip_corner[0] + right_lip_corner[0]) / 2),
        int((left_lip_corner[1] + right_lip_corner[1]) / 2),
    )
    cv2.line(
        vis, left_lip_corner, right_lip_corner, (0, 100, 200), 2
    )  # Line between corners
    cv2.line(
        vis, lip_corner_midpoint, top_lip_middle, (0, 100, 200), 2
    )  # Line to top lip
    cv2.putText(
        vis,
        "top_lip_curvature",
        (lip_corner_midpoint[0] - 60, lip_corner_midpoint[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 100, 200),
        1,
    )
    visualizations["18_top_lip_curvature"] = vis

    # 5. Nose Wrinkle Features
    vis = image.copy()
    nose_top = points[6]
    nose_bottom = points[4]
    cv2.line(vis, nose_top, nose_bottom, (75, 150, 0), 2)
    cv2.putText(
        vis,
        "nose_wrinkle",
        (nose_top[0] - 40, nose_top[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (75, 150, 0),
        1,
    )
    visualizations["19_nose_wrinkle"] = vis

    # 6. Brow Features
    vis = image.copy()
    left_inner_brow = points[65]
    left_outer_brow = points[105]
    right_inner_brow = points[295]
    right_outer_brow = points[334]

    # Brow inner distance
    cv2.line(vis, left_inner_brow, right_inner_brow, (200, 50, 100), 2)
    cv2.putText(
        vis,
        "brow_inner_distance",
        ((left_inner_brow[0] + right_inner_brow[0]) // 2 - 60, left_inner_brow[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 50, 100),
        1,
    )

    # Brow slopes
    cv2.line(vis, left_inner_brow, left_outer_brow, (200, 100, 50), 2)
    cv2.line(vis, right_inner_brow, right_outer_brow, (200, 100, 50), 2)
    cv2.putText(
        vis,
        "left_brow_slope",
        ((left_inner_brow[0] + left_outer_brow[0]) // 2 - 50, left_inner_brow[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 100, 50),
        1,
    )
    cv2.putText(
        vis,
        "right_brow_slope",
        (
            (right_inner_brow[0] + right_outer_brow[0]) // 2 - 50,
            right_inner_brow[1] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 100, 50),
        1,
    )
    visualizations["20_brow_features"] = vis

    # 7. Cheek Features
    vis = image.copy()
    left_cheek = points[117]
    right_cheek = points[346]
    cv2.line(vis, left_cheek, right_cheek, (100, 200, 100), 2)
    cv2.putText(
        vis,
        "cheek_distance",
        (
            (left_cheek[0] + right_cheek[0]) // 2 - 40,
            (left_cheek[1] + right_cheek[1]) // 2 - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 200, 100),
        1,
    )
    visualizations["21_cheek_distance"] = vis

    # 8. Mouth Openness Refined
    vis = image.copy()
    upper_lip_top = points[13]
    lower_lip_bottom = points[14]
    cv2.line(vis, upper_lip_top, lower_lip_bottom, (150, 150, 200), 2)
    cv2.putText(
        vis,
        "vertical_mouth_openness",
        (upper_lip_top[0] - 80, (upper_lip_top[1] + lower_lip_bottom[1]) // 2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 200),
        1,
    )
    visualizations["22_vertical_mouth_openness"] = vis

    # 9. Advanced Eye Features
    vis = image.copy()
    left_eye_top = points[159]
    left_eye_bottom = points[145]
    right_eye_top = points[386]
    right_eye_bottom = points[374]

    cv2.line(vis, left_eye_top, left_eye_bottom, (100, 50, 200), 2)
    cv2.line(vis, right_eye_top, right_eye_bottom, (100, 50, 200), 2)
    cv2.putText(
        vis,
        "left_eye_openness",
        (left_eye_top[0] - 70, left_eye_top[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 50, 200),
        1,
    )
    cv2.putText(
        vis,
        "right_eye_openness",
        (right_eye_top[0] - 70, right_eye_top[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 50, 200),
        1,
    )
    cv2.putText(
        vis,
        "eye_openness_ratio",
        (
            (left_eye_top[0] + right_eye_top[0]) // 2 - 50,
            (left_eye_top[1] + right_eye_top[1]) // 2 - 30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 50, 200),
        1,
    )
    visualizations["23_eye_openness"] = vis

    # 10. Forehead Height
    vis = image.copy()
    forehead_point = points[10]
    brow_mid_point = points[67]
    cv2.line(vis, forehead_point, brow_mid_point, (50, 100, 150), 2)
    cv2.putText(
        vis,
        "forehead_height",
        (forehead_point[0] - 50, (forehead_point[1] + brow_mid_point[1]) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (50, 100, 150),
        1,
    )
    visualizations["24_forehead_height"] = vis

    # Create one complete visualization with all features
    all_features_vis = image.copy()
    for _, vis in visualizations.items():
        # Extract only the colored lines and text (subtract the original image)
        diff = cv2.subtract(vis, image)
        all_features_vis = cv2.add(all_features_vis, diff)

    visualizations["00_all_features"] = all_features_vis

    # Save all visualizations
    for name, vis in visualizations.items():
        output_path = output_dir / f"{name}.jpg"
        cv2.imwrite(str(output_path), vis)

    print(f"All visualizations saved to {output_dir}")

    # Display the all-features visualization
    cv2.imshow("All Facial Features", all_features_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return visualizations


if __name__ == "__main__":
    dataset_path = "dataset"
    try:
        # Visualize examples before processing
        visualize_example_images(dataset_path, num_examples=2)

        single_image_path = r"dataset\Happy\0a50edf08f16feb784999e3d838bcb4aca0202f02aac2aa02fc9f9b4.jpg"
        visualize_all_facial_features(
            single_image_path, "facial_feature_visualizations"
        )

        df = process_dataset(dataset_path)

        # Save to CSV
        output_path = "facial_features22.csv"
        df.to_csv(output_path, index=False)
        print(f"\nFeatures extracted and saved to {output_path}")
        print(f"Total samples in CSV: {len(df)}")
        print("\nFeature columns:")
        for col in df.columns:
            print(f"- {col}")

    except Exception as e:
        print(f"Error: {e}")
