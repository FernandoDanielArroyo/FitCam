import argparse
import cv2
import numpy as np
import mediapipe as mp

def parse():
    parser = argparse.ArgumentParser(
        prog='Yoga FitCam'
    )
    parser.add_argument('-i', '--input_file', default='webcam')
    parser.add_argument('-o', '--output_file', default='No output')
    parser.add_argument('-d', '--display', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse()
    print(f'test poetry script')
    print(args)
    save = False if args.output_file == 'No output' else True 

    # drawing tools
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    Holistic = mp.solutions.holistic.Holistic

    # utility function that helps you to split the different chanels composing an image
    def split_channels(image):
        return (image[..., i] for i in range(image.shape[-1]))

    # utility function that helps you to recompose an image from a list of channels
    def recompose_channels(chan_list):
        return np.stack(chan_list, axis=2).astype(np.uint8)

    def run_filter_with_mediapipe_model(mediapipe_model, mediapipe_based_filter):
        """Run a media pipe model on each video frame grabbed by the webcam and draw results on it

        Args:
            mediapipe_model (): A mediapipe model
            mediapipe_based_filter (): a function to draw model results on frame

        Returns:
            np.ndarray, mediapipe model result
        """
        cap = cv2.VideoCapture(0)
        
        try:
            with mediapipe_model  as model:
                while cap.isOpened():
                    success, image = cap.read()

                    if not success:
                        print("Ignoring empty camera frame.")
                        continue     # If loading a video, use 'break' instead of 'continue'.

                    # Flip the image horizontally for a later selfie-view display, and convert
                    # the BGR image to RGB.
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                    try:
                        results = model.process(image)
                    except Exception:
                        results = None

                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if results and results.pose_landmarks:
                        result_image = mediapipe_based_filter(image, results)
                    else:
                        result_image = image

                    cv2.imshow('MediaPipe', result_image)

        #             if cv2.waitKey(5) & 0xFF == ord('q'):
        #                 break
                        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return image, results

    def draw_holistic_results(image, results, show_hands=True, show_face=True, show_pose=True):
        if show_hands:
            drawing_utils.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
            )

            drawing_utils.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
            )

        if show_face:
            drawing_utils.draw_landmarks(
                image,
                results.face_landmarks,
                mp.solutions.holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_utils.DrawingSpec(thickness=0, circle_radius=0, color=(255, 255, 255)),
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
            )

            # print(results.face_landmarks.landmark)

        if show_pose:
            drawing_utils.draw_landmarks(
                image,
                results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
            )
            
        if results.face_landmarks is not None and results.left_hand_landmarks is not None:
            # print(abs(results.left_hand_landmarks.landmark[8].x - results.face_landmarks.landmark[34].x))
            if abs(results.left_hand_landmarks.landmark[8].x - results.face_landmarks.landmark[34].x) <= 0.13:
                # image = toonify(image)
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
                            output_frame = input_frame.copy()
                            cv2.imshow('MediaPipe', np.array(output_frame)[:,:,::-1])
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

        last_image, last_results = run_filter_with_mediapipe_model(
        mediapipe_model=Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5),
        mediapipe_based_filter=draw_holistic_results
        )   

        return image
    
    
        





