
# ## Bootstrap helper

import cv2
import shutil
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
import tqdm
import argparse

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from fitcam.classes import *

class BootstrapHelper(object):
  """Helps to bootstrap images and filter pose samples for classification."""

  def __init__(self,
               images_in_folder,
               images_out_folder,
               csvs_out_folder):
    self._images_in_folder = images_in_folder
    self._images_out_folder = images_out_folder
    self._csvs_out_folder = csvs_out_folder

    # Get list of pose classes and print image statistics.
    self._pose_class_names = sorted([n for n in os.listdir(self._images_in_folder) if not n.startswith('.')])
    
  def bootstrap(self, per_pose_class_limit=None):
    """Bootstraps images in a given folder.
    
    Required image in folder (same use for image out folder):
      pushups_up/
        image_001.jpg
        image_002.jpg
        ...
      pushups_down/
        image_001.jpg
        image_002.jpg
        ...
      ...

    Produced CSVs out folder:
      pushups_up.csv
      pushups_down.csv

    Produced CSV structure with pose 3D landmarks:
      sample_00001,x1,y1,z1,x2,y2,z2,....
      sample_00002,x1,y1,z1,x2,y2,z2,....
    """
    # Create output folder for CVSs.
    if not os.path.exists(self._csvs_out_folder):
      os.makedirs(self._csvs_out_folder)
      os.makedirs(self._csvs_out_folder+'/ref')
    for pose_class_name in self._pose_class_names:
      print('Bootstrapping ', pose_class_name, file=sys.stderr)

      # Paths for the pose class.
      images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
      images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')
      if not os.path.exists(images_out_folder):
        os.makedirs(images_out_folder)

      with open(csv_out_path, 'w') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        # Get list of images.
        image_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
        if per_pose_class_limit is not None:
          image_names = image_names[:per_pose_class_limit]

        # Bootstrap every image.
        for image_name in tqdm.tqdm(image_names):
          # Load image.
          input_frame = cv2.imread(os.path.join(images_in_folder, image_name))
          input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

          # Initialize fresh pose tracker and run it.
          # with mp_pose.Pose(upper_body_only=False) as pose_tracker:
          with mp_pose.Pose() as pose_tracker:
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

          # Save image with pose prediction (if pose was detected).
          output_frame = input_frame.copy()
          if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
          output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)
          
          # Save landmarks if pose was detected.
          if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array(
                [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                 for lmk in pose_landmarks.landmark],
                dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
            csv_out_writer.writerow([image_name] + pose_landmarks.flatten().astype(str).tolist())

          # Draw XZ projection and concatenate with the image.
          projection_xz = self._draw_xz_projection(
              output_frame=output_frame, pose_landmarks=pose_landmarks)
          output_frame = np.concatenate((output_frame, projection_xz), axis=1)

  def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
    img = Image.new('RGB', (frame_width, frame_height), color='white')

    if pose_landmarks is None:
      return np.asarray(img)

    # Scale radius according to the image width.
    r *= frame_width * 0.01

    draw = ImageDraw.Draw(img)
    for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS:
      # Flip Z and move hips center to the center of the image.
      x1, y1, z1 = pose_landmarks[idx_1] * [1, 1, -1] + [0, 0, frame_height * 0.5]
      x2, y2, z2 = pose_landmarks[idx_2] * [1, 1, -1] + [0, 0, frame_height * 0.5]

      draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
      draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
      draw.line([x1, z1, x2, z2], width=int(r), fill=color)

    return np.asarray(img)

  def align_images_and_csvs(self, print_removed_items=False):
    """Makes sure that image folders and CSVs have the same sample.
    
    Leaves only intersetion of samples in both image folders and CSVs.
    """
    for pose_class_name in self._pose_class_names:
      # Paths for the pose class.
      images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')

      # Read CSV into memory.
      rows = []
      with open(csv_out_path) as csv_out_file:
        csv_out_reader = csv.reader(csv_out_file, delimiter=',')
        for row in csv_out_reader:
          rows.append(row)

      # Image names left in CSV.
      image_names_in_csv = []

      # Re-write the CSV removing lines without corresponding images.
      with open(csv_out_path, 'w') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
          image_name = row[0]
          image_path = os.path.join(images_out_folder, image_name)
          if os.path.exists(image_path):
            image_names_in_csv.append(image_name)
            csv_out_writer.writerow(row)
          elif print_removed_items:
            print('Removed image from CSV: ', image_path)

      # Remove images without corresponding line in CSV.
      for image_name in os.listdir(images_out_folder):
        if image_name not in image_names_in_csv:
          image_path = os.path.join(images_out_folder, image_name)
          os.remove(image_path)
          if print_removed_items:
            print('Removed image from folder: ', image_path)

  def analyze_outliers(self, outliers):
    """Classifies each sample against all other to find outliers.
    
    If sample is classified differrently than the original class - it should
    either be deleted or more similar samples should be added.
    """
    for outlier in outliers:
      image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)

      print('Outlier')
      print('  sample path =    ', image_path)
      print('  sample class =   ', outlier.sample.class_name)
      print('  detected class = ', outlier.detected_class)
      print('  all classes =    ', outlier.all_classes)

      img = cv2.imread(image_path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      show_image(img, figsize=(20, 20))

  def remove_outliers(self, outliers):
    """Removes outliers from the image folders."""
    for outlier in outliers:
      image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)
      os.remove(image_path)

  def print_images_in_statistics(self):
    """Prints statistics from the input image folder."""
    self._print_images_statistics(self._images_in_folder, self._pose_class_names)

  def print_images_out_statistics(self):
    """Prints statistics from the output image folder."""
    self._print_images_statistics(self._images_out_folder, self._pose_class_names)

  def _print_images_statistics(self, images_folder, pose_class_names):
    print('Number of images per pose class:')
    for pose_class_name in pose_class_names:
      n_images = len([
          n for n in os.listdir(os.path.join(images_folder, pose_class_name))
          if not n.startswith('.')])
      print('  {}: {}'.format(pose_class_name, n_images))


# ## Bootstrap images


def parse_preprocess():
    parser = argparse.ArgumentParser(
        prog='preprocess'
    )
    parser.add_argument('-a', '--auto_mode', action='store_true')
    args = parser.parse_args()
    return args

def preprocess():
  args = parse_preprocess()
  print('parse args')
  # Required structure of the images_in_folder:
  #
  #   fitness_poses_images_in/
  #     pushups_up/
  #       image_001.jpg
  #       image_002.jpg
  #       ...
  #     pushups_down/
  #       image_001.jpg
  #       image_002.jpg
  #       ...
  #     ...

  # Output folders for bootstrapped images and CSVs.

  bootstrap_images_in_folder = 'yoga_images_in'
  bootstrap_images_out_folder = 'yoga_images_out'
  bootstrap_csvs_out_folder = 'data'

  
  # Initialize helper.
  bootstrap_helper = BootstrapHelper(
      images_in_folder=bootstrap_images_in_folder,
      images_out_folder=bootstrap_images_out_folder,
      csvs_out_folder=bootstrap_csvs_out_folder,
  )

  # Check how many pose classes and images for them are available.
  bootstrap_helper.print_images_in_statistics()

  # Bootstrap all images.
  # Set limit to some small number for debug.
  bootstrap_helper.bootstrap(per_pose_class_limit=None)

  # Check how many images were bootstrapped.
  bootstrap_helper.print_images_out_statistics()

  # After initial bootstrapping images without detected poses were still saved in
  # the folderd (but not in the CSVs) for debug purpose. Let's remove them.
  bootstrap_helper.align_images_and_csvs(print_removed_items=False)
  bootstrap_helper.print_images_out_statistics()

  print('auto mode or not')

  if args.auto_mode: 
    # ## Automatic filtration
    # 
    # Classify each sample against database of all other samples and check if it gets in the same class as annotated after classification.
    # 
    # There can be two reasons for the outliers:
    # 
    #   * **Wrong pose prediction**: In this case remove such outliers.
    # 
    #   * **Wrong classification** (i.e. pose is predicted correctly and you aggree with original pose class assigned to the sample): In this case sample is from the underrepresented group (e.g. unusual angle or just very few samples). Add more similar samples and run bootstrapping from the very beginning.
    # 
    # Even if you just removed some samples it makes sence to re-run automatic filtration one more time as database of poses has changed.
    # 
    # **Important!!** Check that you are using the same parameters when classifying whole videos later.

    # Find outliers.

    # Transforms pose landmarks into embedding.
    pose_embedder = FullBodyPoseEmbedder()

    # Classifies give pose against database of poses.
    pose_classifier = PoseClassifier(
        pose_samples_folder=bootstrap_csvs_out_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    outliers = pose_classifier.find_pose_sample_outliers()
    print('Number of outliers: ', len(outliers))

    # Analyze outliers.
    bootstrap_helper.analyze_outliers(outliers)

    # Remove all outliers (if you don't want to manually pick).
    bootstrap_helper.remove_outliers(outliers)

    # Align CSVs with images after removing outliers.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

  else:
    # ## Manual filtration
    # 
    # Please manually verify predictions and remove samples (images) that has wrong pose prediction. Check as if you were asked to classify pose just from predicted landmarks. If you can't - remove it.
    # 
    # Align CSVs and image folders once you are done.

    # Align CSVs with filtered images.
    input('Mettz Yes si vous avez fini')
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

  # Move samples image to data/ref

  path_images_process = Path('yoga_images_in')
  path_ref_images = Path('data/ref')

  for csv in Path('data/').glob('*.csv'):
    pose_name = csv.stem
    out_image_name = path_ref_images / f'{pose_name}.png'
    if not out_image_name.exists():
      # select a pose in the path_images_process:
      path_images_pose = path_images_process / pose_name
      print(path_images_pose)
      images = list(path_images_pose.glob('*'))
      print(images)
      shutil.copy(images[0], out_image_name)


if __name__ == '__main__':
  print('Test preprocessing')
