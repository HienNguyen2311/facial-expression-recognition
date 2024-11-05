# Functions for Preprocessing and Handling Image Data
1. `get_annotations`: Retrieves annotation file paths and image names from a specified directory.
2. `load_image_labels`: Loads arousal, valence, and expression labels for each image.
3. `show_original_image`: Displays a random selection of original images.
4. `separate_labels`: Separates arousal, valence, and expression labels into separate lists.
5. `get_img_rgb`: Extracts RGB values from images.
6. `create_labels_data_df`: Creates a pandas DataFrame with image names, RGB values, and labels.
7. `count_exp_labels`: Counts the number of unique labels in a specified column.
8. `sampling_images`: Samples a specified number of images from each expression category.
9. `keep_original_face`: Returns the original image without modification.
10. `face_cropping`: Crops and converts a face image to grayscale.
11. `img_preprocessing`: Performs face detection, eye detection, face alignment, and optional histogram equalization and noise reduction.
12. `resize_convert_images`: Resizes and converts images to a specified size.
13. `save_dataset_to_hp5`: Saves preprocessed images and labels to an HDF5 file.
14. `show_preprocessed_images`: Displays a selection of preprocessed images.
15. `load_dataset_from_disk`: Loads images and labels from an HDF5 file.
16. `load_resize`: Loads and resizes images from an HDF5 file.
17. `load_checkpoint`: Loads a checkpoint file using pickle.