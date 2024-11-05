import dill as pickle
from PIL import Image
import random
from matplotlib import pyplot as plt
from IPython.display import display
import numpy as np
import math, os, pdb, cv2, copy, time, dlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
import sys
import h5py
import torch
import torchvision.transforms as transforms

def get_annotations(path_anno, annotation_path_names, annotation_filenames, annotation_image_names):
    for folder, subfolders, filenames in os.walk(path_anno):
        for anno in filenames:
            annotation_path_names.append(folder+'/'+anno)
            annotation_filenames.append(anno)

    for filenames in annotation_filenames:
        image_name = filenames.split('_')[0]
        annotation_image_names.append(image_name)

    unique_annotation_image_names = list(set(annotation_image_names))

#     print("unique image names:", unique_annotation_image_names)
#     print("image names:", annotation_image_names)
#     print("file names:", annotation_filenames)
#     print("path names:", annotation_path_names)
    print("numbers of anno files", len(annotation_path_names))
    print("numbers of unique images", len(unique_annotation_image_names))

    return unique_annotation_image_names, annotation_image_names, annotation_filenames, annotation_path_names

def load_image_labels(path_anno, unique_annotation_image_names, all_image_values):
    for image_name in unique_annotation_image_names:
        image_values = []
        for label in ("aro", "val", "exp"):
            file_name = f"{image_name}_{label}.npy"
            file_path = os.path.join(path_anno, file_name)
            value = np.load(file_path)
            image_values.append(value)

        all_image_values.append(image_values)

    print("numbers of images:", len(all_image_values))

    return all_image_values

def show_original_image(list_image_path, unique_annotation_image_names, data_path):
    for index, image_name in enumerate(unique_annotation_image_names):
        list_image_path.append(data_path+'/'+image_name+'.jpg')


    columns = 30
    rows = 1
    count = columns * rows

    randomlist = random.sample(range(0, len(list_image_path)), count)

    plt.figure(figsize=(200,200))
    for i, idx in enumerate(randomlist):
        img = Image.open(list_image_path[idx])
        plt.subplot(rows, columns, i + 1)
        plt.imshow(img)
        plt.axis('off')

    return list_image_path

def separate_labels(aro_label_values, val_label_values, exp_label_values, all_image_values):
    aro_label_values = []
    val_label_values = []
    exp_label_values = []

    for label_set in all_image_values:
        aro_label_values.append(label_set[0])
        val_label_values.append(label_set[1])
        exp_label_values.append(label_set[2])

    print(len(aro_label_values))
    print(len(val_label_values))
    print(len(exp_label_values))

    return aro_label_values, val_label_values, exp_label_values

def get_img_rgb(red, green, blue, img_paths):
    for i in range(len(img_paths)):
        img = Image.open(img_paths[i])
        rgb_image = img.convert('RGB')
        r, g, b = rgb_image.getpixel((0, 0))
        red.append(r)
        green.append(g)
        blue.append(b)
    return red, green, blue

def create_labels_data_df(unique_annotation_image_names, aro_label_values, val_label_values, exp_label_values,
                          rgb = True, red = None, green = None, blue = None):
    if rgb == True:
        df = pd.DataFrame({
            'imagename': unique_annotation_image_names,
            'red': red,
            'green': green,
            'blue': blue,
            'aro': aro_label_values,
            'val': val_label_values,
            'exp': exp_label_values
        })
    else:
        df = pd.DataFrame({
            'imagename': unique_annotation_image_names,
            'aro': aro_label_values,
            'val': val_label_values,
            'exp': exp_label_values
        })  

    return df

def count_exp_labels(df, column):

    # Count the number of unique labels in the 'label' column
    label_counts = df[column].value_counts()
    print(label_counts)

def sampling_images(df, number_of_samples):
    sampled_images = (df.groupby('exp')['imagename'].apply(lambda s: s.sample(min(len(s), number_of_samples), random_state = 42))).astype(str)
    exp_image_lists = {}
    for exp, group in sampled_images.groupby(level=0):
        exp_image_lists[exp] = group.tolist()
    chosen_train_images = []
    chosen_train_images = [img for exp_list in exp_image_lists.values() for img in exp_list]
    random.shuffle(chosen_train_images)
    print(len(chosen_train_images))
    return chosen_train_images

def keep_original_face(original_image):
  face2 = original_image
  #cv2_imshow(face2)
  return face2

def face_cropping(roi):
  face = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
  face2 = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
  #cv2_imshow(face2)
  return(face2)

def img_preprocessing(network, eye_cascade, image_data, complex_images, histogram_equalizer, noise_reduction):
    #print(image_data)
    image = Image.open(image_data).convert('L')
    image_np = np.array(image, 'uint8')
    # cv2_imshow(image_np)
    # print("change from color scale to gray scale", image_np.shape)


    image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (100, 100)), 1.0, (100,100), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    roi = []
    conf_min = 0.5
    image_cp = image.copy()
    for i in range(0, detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence > conf_min:
        bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = bbox.astype('int')
        roi = image_cp[start_y:end_y, start_x:end_x]
        #text = "{:.2f}%".format(confidence * 100)
        #cv2.putText(image, text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)
        #cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
    # cv2_imshow(image)
    # print("face detection with bounding box but the box is comment", image.shape)
    if len(roi) == 0:
      face2 = keep_original_face(image_np)
      #print("face without cropping due to face detection fail", face2.shape)
      complex_images.append(image_data)

    else:

      # Creating two regions of interest
      roi_gray=image[start_y:end_y, start_x:end_x]
      roi_color=image[start_y:end_y, start_x:end_x]
      # Creating variable eyes
      eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
      index=0
      # Creating for loop in order to divide one eye from another
      for (ex , ey,  ew,  eh) in eyes:
        if index == 0:
          eye_1 = (ex, ey, ew, eh)
        elif index == 1:
          eye_2 = (ex, ey, ew, eh)
      # Drawing rectangles around the eyes
        # cv2.rectangle(roi_color, (ex,ey) ,(ex+ew, ey+eh), (0,0,255), 3)
        index = index + 1
      #cv2_imshow(image)

      try:
        eye_1[0] < eye_2[0]
      except UnboundLocalError as e:
          try:
            face2 = face_cropping(roi)
            #print("face cropping only due to no eye detected", face2.shape)
            complex_images.append(image_data)
            if face2.shape[0] < 100 or face2.shape[1] < 100:
              face2 = keep_original_face(image_np)
              #print("face without cropping due to croping fail", face2.shape)
              complex_images.append(image_data)
          except cv2.error as e:
              face2 = keep_original_face(image_np)
              #print("face without cropping due to roi empty", face2.shape)
              complex_images.append(image_data)
      else:
        if eye_1[0] < eye_2[0]:
          left_eye = eye_1
          right_eye = eye_2
        else:
          left_eye = eye_2
          right_eye = eye_1

        # Calculating coordinates of a central points of the rectangles
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]

        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]

        #cv2.circle(roi_color, left_eye_center, 5, (255, 0, 0) , -1)
        #cv2.circle(roi_color, right_eye_center, 5, (255, 0, 0) , -1)
        #cv2.line(roi_color,right_eye_center, left_eye_center,(0,200,200),3)
        if left_eye_y > right_eye_y:
          A = (right_eye_x, left_eye_y)
          # Integer -1 indicates that the image will rotate in the clockwise direction
          direction = -1
        else:
          A = (left_eye_x, right_eye_y)
          # Integer 1 indicates that image will rotate in the counter clockwise
          # direction
          direction = 1

        #cv2.circle(roi_color, A, 5, (255, 0, 0) , -1)

        # cv2.line(roi_color,right_eye_center, left_eye_center,(0,200,200),3)
        # cv2.line(roi_color,left_eye_center, A,(0,200,200),3)
        # cv2.line(roi_color,right_eye_center, A,(0,200,200),3)
        #cv2_imshow(image)

        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        if delta_x == 0 or delta_y == 0:
          face2 = face_cropping(roi)
          #print("face cropping only due to wrong eye detected", face2.shape)
          if face2.shape[0] < 100 or face2.shape[1] < 100:
            face2 = keep_original_face(image_np)
            #print("face without cropping due to croping fail", face2.shape)
            complex_images.append(image_data)

        else:

          angle = np.arctan(delta_y/delta_x)
          angle = (angle * 180) / np.pi
          if angle > 20 or angle < -20:
            face2 = face_cropping(roi)
            #print("face cropping only due to wrong eye detected", face2.shape)
            if face2.shape[0] < 100 or face2.shape[1] < 100:
              face2 = keep_original_face(image_np)
              #print("face without cropping due to croping fail", face2.shape)
              complex_images.append(image_data)
          else:

            # Width and height of the image
            h, w = image.shape[:2]
            # Calculating a center point of the image
            # Integer division "//"" ensures that we receive whole numbers
            center = (w // 2, h // 2)
            # Defining a matrix M and calling
            # cv2.getRotationMatrix2D method
            M = cv2.getRotationMatrix2D(center, (angle), 1.0)

            # Applying the rotation to our image using the
            # cv2.warpAffine method
            rotated = cv2.warpAffine(image, M, (w, h))
            #cv2_imshow(rotated)
            #print("face alignment", rotated.shape)

            roi = rotated[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            face2 = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            #cv2_imshow(face2)
            #print("face cropped according to the bounding box", face2.shape)
            if face2.shape[0] < 100 or face2.shape[1] < 100:
              face2 = keep_original_face(image_np)
              #print("face without cropping due to croping fail", face2.shape)
              complex_images.append(image_data)

    if histogram_equalizer:
        face2 = cv2.equalizeHist(face2)
        #cv2_imshow(face2)
        #print("histogram equalizer face")

    if noise_reduction:
        # non-local means denoising
        face2 = cv2.fastNlMeansDenoising(face2, None, 10, 10, 7)
        #cv2_imshow(face2)
        #print("noise reduction face")

    return face2

def resize_convert_images(image_size, images_data):
    resized_images_list = []
    for image in images_data:
        gray_image = tf.expand_dims(image, axis=-1)
        resized_image = tf.image.resize(gray_image, image_size)
        resized_images_list.append(resized_image)

    image_batch = tf.stack(resized_images_list, axis=0)

    # Convert the tensor to a numpy array
    new_image_batch = image_batch.numpy()

    return new_image_batch

def save_dataset_to_hp5(aro_label_values, val_label_values, exp_label_values, file_name, X):
    av_labels = np.stack((aro_label_values, val_label_values), axis=-1)

    # Save images and labels to an HDF5 file
    with h5py.File(file_name, 'w') as h5f:
        h5f.create_dataset('imagesvalue', data = X)
        h5f.create_dataset('av', data = np.array(av_labels).astype(float))
        h5f.create_dataset('aro', data = np.array(aro_label_values).astype(float))
        h5f.create_dataset('val', data = np.array(val_label_values).astype(float))
        h5f.create_dataset('exp', data = np.array(exp_label_values).astype(int))

def show_preprocessed_images(X, number_of_images_shown):
    for i, image in enumerate(X[:number_of_images_shown]):
    # Convert the image tensor to a numpy array
    # If the image is in a tensor format, make sure to run .numpy() to convert it
    # image = image.numpy()

    # If your image data is in the range [0, 1], you can display it directly.
    # If it's in the range [0, 255], you may need to scale it by dividing by 255.

    # Ensure the image data is float, which might not be necessary depending on your TensorFlow version
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        plt.figure()
        plt.imshow(image, cmap='gray')  # Use cmap='gray' for grayscale images
        plt.title(f'Image {i}')
        plt.axis('off')  # Turn off the axis
        plt.show()

def load_dataset_from_disk(data_file_name):
    with h5py.File(data_file_name, 'r') as h5f:
        # Load the images
        X = h5f['imagesvalue'][:]
        y = h5f['av'][:]  # labels will have shape [num_samples, 2]
    return X, y

def load_resize(image_size, base_data_file_name, save_file_name):
    X, y = load_dataset_from_disk(base_data_file_name)
    # Convert NumPy array to TensorFlow tensor
    X_tensor = tf.convert_to_tensor(X)
    # Resize the images
    resized_images = tf.image.resize(X_tensor, image_size)
    # Convert back to NumPy array if needed
    resized_X = resized_images.numpy()
    return resized_X

# Function to load a checkpoint
def load_checkpoint(filename):
    with open(filename, "rb") as cp_file:
        cp = pickle.load(cp_file)
    return cp