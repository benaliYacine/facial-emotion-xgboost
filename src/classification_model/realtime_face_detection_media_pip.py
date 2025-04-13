import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open the camera
cap = cv2.VideoCapture(0)  # 0 is usually the built-in camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Press 'q' to quit")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame from camera. Exiting...")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect faces
    results = face_mesh.process(rgb_frame)

    # Count detected faces
    num_faces = 0
    if results.multi_face_landmarks:
        num_faces = len(results.multi_face_landmarks)

        # Draw face landmarks for each detected face
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh
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

    # Display the number of detected faces
    cv2.putText(
        frame,
        f"Visages: {num_faces}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # Display the frame with detections
    cv2.imshow("Real-time Face Mesh Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
