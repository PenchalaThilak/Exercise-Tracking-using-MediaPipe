import cv2
import mediapipe as mp
import numpy as np
import gradio as gr

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variables for rep counting
counter = 0
stage = None

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to process a single frame
def process_frame(frame):
    global counter, stage
    if frame is None:
        return None, f"Reps: {counter}", f"Stage: {stage if stage else 'None'}"

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process frame with MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angle = calculate_angle(shoulder, elbow, wrist)

        # Display angle on the frame
        cv2.putText(image, str(round(angle, 2)),
                    tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Rep counting logic
        if angle > 160:
            stage = "down"
        if angle < 30 and stage == "down":
            stage = "up"
            counter += 1

        # Display reps and stage
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage if stage else "", (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    except:
        pass

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, f"Reps: {counter}", f"Stage: {stage if stage else 'None'}"

# Function to process webcam stream using Gradio's webcam input
def process_webcam_stream(frame):
    global counter, stage
    counter = 0  # Reset counter for new stream
    stage = None
    return process_frame(frame)

# Function to process uploaded video
def process_uploaded_video(video_path):
    global counter, stage
    counter = 0
    stage = None
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb, reps, stage = process_frame(frame)
        yield image_rgb, reps, stage
    cap.release()

# Gradio interface using gr.Blocks
with gr.Blocks() as demo:
    gr.Markdown("# ExerciseSense: AI-Powered Curl Detection")
    gr.Markdown("Choose an input source to track bicep curls and pose.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_source = gr.Dropdown(choices=["Use Webcam", "Upload Video"], label="Select Input Source", value="Use Webcam", elem_classes="small-dropdown")
            webcam_input = gr.Image(label="Webcam Feed", sources=["webcam"], streaming=True, visible=True, elem_classes="small-video-input")
            video_input = gr.Video(label="Upload Video", visible=False, elem_classes="small-video-input")
        
        with gr.Column(scale=3):
            with gr.Row():
                video_output = gr.Image(label="Pose Detection Feed", streaming=True, elem_classes="large-video")
                with gr.Column():
                    rep_count = gr.Textbox(label="Rep Count", elem_classes="small-textbox")
                    stage_output = gr.Textbox(label="Stage", elem_classes="small-textbox")

    # Custom CSS for styling
    demo.css = """
    .small-dropdown { max-width: 200px !important; }
    .small-button { max-width: 200px !important; padding: 5px !important; font-size: 14px !important; }
    .small-video-input { max-width: 200px !important; }
    .large-video { width: 100% !important; max-width: 800px !important; margin: 0 auto !important; }
    .small-textbox { max-width: 150px !important; height: 50px !important; margin: 10px 0 !important; }
    """

    # Function to toggle visibility of input components based on dropdown
    def toggle_inputs(source):
        if source == "Use Webcam":
            return gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None)
        else:
            return gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None), gr.update(value=None)

    # Update visibility and clear outputs when dropdown changes
    input_source.change(
        toggle_inputs,
        inputs=[input_source],
        outputs=[webcam_input, video_input, video_output, rep_count, stage_output]
    )

    # Process webcam stream when a new frame is received
    webcam_input.stream(
        process_webcam_stream,
        inputs=[webcam_input],
        outputs=[video_output, rep_count, stage_output],
        queue=True
    )

    # Process uploaded video when a file is uploaded
    video_input.change(
        process_uploaded_video,
        inputs=[video_input],
        outputs=[video_output, rep_count, stage_output],
        queue=True
    )

demo.launch()
