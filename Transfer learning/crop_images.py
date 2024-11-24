import face_recognition
from PIL import Image
import os

def process_faces(input_root, output_root, size=(128, 128)):
    """
    Detect faces in images, crop them, resize to 128x128, and save.

    Args:
        input_root (str): Root folder of the dataset.
        output_root (str): Root folder to save processed images.
        size (tuple): Output image size (width, height).
    """
    for root, dirs, files in os.walk(input_root):
        # Maintain folder structure
        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)

        # Create output folder if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Load image with face_recognition
                    input_path = os.path.join(root, file)
                    image = face_recognition.load_image_file(input_path)

                    # Find face locations
                    face_locations = face_recognition.face_locations(image)

                    if not face_locations:
                        print(f"No face detected in {file}. Skipping...")
                        continue

                    for i, face_location in enumerate(face_locations):
                        # Get the coordinates of the face
                        top, right, bottom, left = face_location

                        # Crop the face
                        face_image = image[top:bottom, left:right]
                        pil_image = Image.fromarray(face_image)

                        # Resize to the desired size
                        pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)

                        # Save the face
                        output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_face{i+1}.jpg")
                        pil_image.save(output_file)

                    print(f"Processed faces in {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")


# Example usage
input_folder = "./dataset"  # Replace with your dataset folder path
output_folder = "./dataset2"  # Replace with your desired output folder

# process_faces(input_folder, output_folder)


# convert images into 1288128 size
import os   
from PIL import Image

def resize_images(input_root, output_root, size=(256, 256)):
    """
    Resize images to the desired size.

    Args:
        input_root (str): Root folder of the dataset.
        output_root (str): Root folder to save resized images.
        size (tuple): Output image size (width, height).
    """
    for root, dirs, files in os.walk(input_root):
        # Maintain folder structure
        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Load image
                    input_path = os.path.join(root, file)
                    image = Image.open(input_path)
                    # Resize to the desired size
                    resized_image = image.resize(size, Image.LANCZOS)
                    # Save the resized image
                    output_file = os.path.join(output_dir, file)
                    resized_image.save(output_file)
                    print(f"Resized {file}")
                except Exception as e:
                    print(f"Error resizing {file}: {e}")

# Example usage
input_folder = "./dataset2"  # Replace with your dataset folder path
output_folder = "./dataset2"  # Replace with your desired output folder

resize_images(input_folder, output_folder)
