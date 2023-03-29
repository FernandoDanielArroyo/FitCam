import os
import numpy as np
import tqdm
import cv2
from classes import FullBodyPoseEmbedder
from classes import PoseClassifier
from classes import EMADictSmoothing
from classes import RepetitionCounter
from classes import PoseClassificationVisualizer
from classes import show_image
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing


video_path = 'pushups.mp4'
class_name='pushups_up'
out_video_path = 'pushups-sample-out.mp4'

video_cap = cv2.VideoCapture(video_path)

# Get some video parameters to generate output video with classificaiton.
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"video_n_frames: {video_n_frames}")
print(f"video_fps: {video_fps}")
print(f"video_width: {video_width}")
print(f"video_height: {video_height}")

# Initilize tracker, classifier and counter.
# Do that before every video as all of them have state.
from mediapipe.python.solutions import pose as mp_pose


# Folder with pose class CSVs. That should be the same folder you using while
# building classifier to output CSVs.
pose_samples_folder = 'fitness_poses_csvs_out'

# Initialize tracker.
pose_tracker = mp_pose.Pose()

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Initialize classifier.
# Ceck that you are using the same parameters as during bootstrapping.
pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# # Uncomment to validate target poses used by classifier and find outliers.
# outliers = pose_classifier.find_pose_sample_outliers()
# print('Number of pose sample outliers (consider removing them): ', len(outliers))

# Initialize EMA smoothing.
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# Initialize counter.
repetition_counter = RepetitionCounter(
    class_name=class_name,
    enter_threshold=6,
    exit_threshold=4)

# Initialize renderer.
pose_classification_visualizer = PoseClassificationVisualizer(
    class_name=class_name,
    plot_x_max=1000,
    # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
    plot_y_max=10)


cap = cv2.VideoCapture(0)
print(cap.isOpened())
try:
    while cap.isOpened():
        success, input_frame = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            break #continue     # If loading a video, use 'break' instead of 'continue'.

        # Run pose tracker.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        #Draw pose prediction.
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
          mp_drawing.draw_landmarks(
              image=output_frame,
              landmark_list=pose_landmarks,
              connections=mp_pose.POSE_CONNECTIONS)

        if pose_landmarks is not None:
          # Get landmarks.
          frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
          pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                      for lmk in pose_landmarks.landmark], dtype=np.float32)
          assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

          # Classify the pose on the current frame.
          pose_classification = pose_classifier(pose_landmarks)

          # Smooth classification using EMA.
          pose_classification_filtered = pose_classification_filter(pose_classification)

          # Count repetitions.
          repetitions_count = repetition_counter(pose_classification_filtered)
        else:
          # No pose => no classification on current frame.
          pose_classification = None

          # Still add empty classification to the filter to maintaing correct
          # smoothing for future frames.
          pose_classification_filtered = pose_classification_filter(dict())
          pose_classification_filtered = None

          # Don't update the counter presuming that person is 'frozen'. Just
          # take the latest repetitions count.
          repetitions_count = repetition_counter.n_repeats

        # Draw classification plot and repetition counter.

        #fps = cap.get(cv2.CAP_PROP_FPS)
        #print('fps :',fps)  # 30 images par secondes
        output_frame = pose_classification_visualizer(
        frame=output_frame,
        pose_classification=pose_classification,
        pose_classification_filtered=pose_classification_filtered,
        Timer=0, # Nb de Frame
        fps = 30, #Nb de frame en une seconde. Possiblement a revérifier abc les 2 commandes fps commentés ci-dessus
        )
        
        cv2.imshow('MediaPipe', np.array(output_frame)[:,:,::-1])

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()     