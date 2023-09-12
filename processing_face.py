
from imgaug import augmenters as iaa
import cv2

def rotate_image(original_image, angle):
    
    # Get the height and width of the original image
    height, width = original_image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Apply the rotation to create a new image
    rotated_image = cv2.warpAffine(original_image, rotation_matrix, (width, height))

    return rotated_image

def transform_face(face):
    list_face = []
    list_face.append(face)
    
    list_face.append(rotate_image(face, 10))
    
    list_face.append(rotate_image(face, -10))
    
    augmentation = iaa.Sequential([
    iaa.Fliplr(1),  # Random rotation between -45 and 45 degrees
    ])
    list_face.append(augmentation(image = face))
    
    return list_face