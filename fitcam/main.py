import argparse
import cv2
import numpy as np

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
        





