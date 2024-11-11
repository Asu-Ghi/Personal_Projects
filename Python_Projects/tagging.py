import cv2 as cv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions


# set working directory
directory = "/Users/asughimire/Desktop/Junior Year/Data Matters/Homework/Homework_11/images/lec__4____highLevelImage_wLab_export/"

# Load MobileNetV2 pre-trained model
model = MobileNetV2(weights='/Users/asughimire/Desktop/Personal_Projects/Personal_Projects/Python_Projects/Weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')

# load images from file path
def load_images():
    image_paths = []
    # open all image path names
    for filename in os.listdir(directory):
        if(filename.lower().endswith(".jpeg")):
            image_paths.append(filename)

    # Sort by characters 3-4 
    image_paths.sort(key = lambda x: int(x[2: -5]))

    # add directory back to paths
    for i in range(len(image_paths)):
        image_paths[i] = directory + image_paths[i]

    # Stores cv2 images
    images = []

    # Load all images
    for image_path in image_paths:
        img = cv.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
        else:
            images.append(img)

    # return images
    return images

# display images in a grid
def grid_display_images(images):

    # display image constants
    SCREEN_WIDTH = 1900
    SCREEN_HEIGHT = 1060
    n_rows = 3
    n_cols = 10

    # Resize images
    resized_images = []

    fixed_width = SCREEN_WIDTH // n_cols  # Example: three images per row
    fixed_height = SCREEN_HEIGHT // n_rows  # Example: three images per column

    for img in images:
        # Get the original dimensions of the image
        original_height, original_width = img.shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height

        # Resize by width or height based on which dimension needs to be constrained
        if original_width > original_height:  # Landscape or wide image
            new_width = fixed_width
            new_height = int(new_width / aspect_ratio)
            if new_height > fixed_height:
                new_height = fixed_height
                new_width = int(new_height * aspect_ratio)
        else:  # Portrait or square image
            new_height = fixed_height
            new_width = int(new_height * aspect_ratio)
            if new_width > fixed_width:
                new_width = fixed_width
                new_height = int(new_width / aspect_ratio)
        
        # Resize the image while maintaining its aspect ratio
        resized_img = cv.resize(img, (new_width, new_height))
        
        # Add padding if necessary to fit the fixed dimensions
        top_padding = (fixed_height - new_height) // 2
        bottom_padding = fixed_height - new_height - top_padding
        left_padding = (fixed_width - new_width) // 2
        right_padding = fixed_width - new_width - left_padding
        
        # Pad the resized image to fit the fixed dimensions
        padded_img = cv.copyMakeBorder(resized_img, top_padding, bottom_padding, left_padding, right_padding,
                                    cv.BORDER_CONSTANT, value=(0, 0, 0))  # Black padding
        
        resized_images.append(padded_img)


    grid = []

    # Ensure np.hstack() is working by checking the images in each row
    for i in range(0, len(resized_images), n_cols):
        # Check if all images in the row have the same shape
        row_images = resized_images[i:i + n_cols]
        
        # If the row has fewer images than expected, pad with empty (black) images
        if len(row_images) < n_cols:
            # Pad with black images (all zeros)
            pad_image = np.zeros((fixed_height, fixed_width, 3), dtype=np.uint8)  # Blank black image
            while len(row_images) < n_cols:
                row_images.append(pad_image)  # Add blank image to make the row complete

        # Check shapes of images in the current row
        row_shapes = [img.shape for img in row_images]
        print(f"Row {i // n_cols} shapes: {row_shapes}")
        
        # If shapes are consistent, proceed with horizontal stacking
        if len(set(row_shapes)) == 1:
            grid.append(np.hstack(row_images))
        else:
            print(f"Skipping row {i // n_cols} due to shape mismatch.")


    # vertically stack the columns
    display_img = np.vstack(grid)


    # display image
    cv.imshow('Image Grid', display_img)
    cv.waitKey(0)
    cv.destroyAllWindows


# tag image
def mobile_net_tag(image):
    # preprocess image for mobile net
    img_resized = cv.resize(image, (224, 224)) # Resize to 224 x 224
    img_array = np.expand_dims(img_resized, axis=0) # Adds batch dim to 3d image data -> (batch, height, width, rgb)
    img_preprocessed = preprocess_input(img_array) #  Preprocess image for MobileNetV2

    # Get predictions from the model
    preds = model.predict(img_preprocessed)
        
    # Decode the predictions to get human-readable labels
    decoded_preds = decode_predictions(preds, top=3)[0]  # Get top 3 predictions
    
    # Initialize an empty tag list for the current image
    current_tags = []
    
    # Check the top predictions and map them to predefined tags
    for _, label, _ in decoded_preds:
        for tag, keywords in tags.items():
            if any(keywords.lower() in label.lower() for keywords in keywords):
                current_tags.append(tag)
    
    return current_tags



# Stores tags for each image
image_tags = []

# Set tags
tags = {
    "mountain": ["alp", "mountain", "hill"],
    "tree": ["tree", "forest", "plant"],
    "prettySky": ["sky", "sunset", "sunrise"],
    "person": ["person", "man", "woman"],
    "child": ["child", "baby"],
    "animal": ["dog", "cat", "animal", "bird"],
    "virginiaTech": ["school", "university", "campus"],
    "food": ["food", "meal", "dish"],
    "entertainment": ["stage", "concert", "theater"],
    "city": ["city", "urban"],
    "country": ["countryside", "rural"],
    "building": ["building", "house", "architecture"]
}

images = load_images()

for image in images:
    image_tags.append(mobile_net_tag(image))


for i, tags in enumerate(image_tags):
    print(f"Image {[i]} is tagged with: {', '.join(tags)}")









