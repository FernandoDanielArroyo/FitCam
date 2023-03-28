import argparse
import cv2
import numpy as np
from pathlib import Path
from mediapipe.python.solutions import pose as mp_pose
from fitcam.classes import *
from mediapipe.python.solutions import drawing_utils as mp_drawing

def parse():
    parser = argparse.ArgumentParser(
        prog='Yoga FitCam'
    )
    parser.add_argument('-i', '--input_file', default='webcam')
    parser.add_argument('-o', '--output_file', default='No output')
    parser.add_argument('-d', '--display', action='store_true')
    args = parser.parse_args()
    return args

def load_poses():
    path = Path('data')
    poses = []
    for csv_file in path.glob('*.csv'):
        poses.append(csv_file.stem)
    return poses

def main():
    poses = load_poses()
    print(poses)
    args = parse()
    print(f'FitCam Yoga v0.25')
    print(args)
    save = False if args.output_file == 'No output' else True 
    class_name = 'warrior2'
    pose_samples_folder = 'data'

    pose_tracker = mp_pose.Pose()
    pose_embedder = FullBodyPoseEmbedder()

    pose_classifier = PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    pose_classification_filter = EMADictSmoothing(
        window_size=10,
        alpha=0.2)

    pose_classification_visualizer = PoseClassificationVisualizer(
        class_name=class_name,
        plot_x_max=1000,
        # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
        plot_y_max=10)

    pose_classification = None
    pose_landsmarks = None

    if args.input_file == 'webcam':
        print('Loading WebCam')
        cap = cv2.VideoCapture(0)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if save:
            out_video = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

        try:
            while cap.isOpened():

                success, input_frame = cap.read()
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
                    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                    pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                    for lmk in pose_landmarks.landmark], dtype=np.float32)
                    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                    # Classify the pose on the current frame.
                    pose_classification = pose_classifier(pose_landmarks)
                    pose_classification_filtered = pose_classification_filter(pose_classification)
                else:
                    # No pose => no classification on current frame.
                    pose_classification = None

                    # Still add empty classification to the filter to maintaing correct
                    # smoothing for future frames.
                    pose_classification_filtered = pose_classification_filter(dict())
                    pose_classification_filtered = None

                    # Don't update the counter presuming that person is 'frozen'. Just
                    # take the latest repetitions count.

                print(pose_classification)
                if pose_classification:
                    list_val = [(v, k) for k,v in pose_classification.items()]
                    sorted_list = sorted(list_val, reverse=True)

                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                if pose_classification:
                    output_frame = cv2.putText(output_frame, f'{sorted_list[0]}', org=(100,100), fontFace=1, fontScale=1, color=1)
                cv2.imshow('MediaPipe', output_frame)

                if save:
                    out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()   
    else:
        print('Loading Video')
        try:
            cap = cv2.VideoCapture(args.input_file)
            frame_id = 0
            video_n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if save:
                out_video = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

            for i in range(int(video_n_frames)):
                success, input_frame = cap.read()
                output_frame = input_frame.copy()
                cv2.imshow('MediaPipe', np.array(output_frame)[:,:,::-1])
                if cv2.waitKey(5) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()   
        





