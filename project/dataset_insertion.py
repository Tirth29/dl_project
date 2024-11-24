import os
import pandas as pd
import gdown

# Google Drive link for the Excel file
excel_file_url = "https://docs.google.com/spreadsheets/d/1ZYZlEvWqDLbIA6Ktg7NsWRRfy64QiBOYIqMSf-a44L0/edit?usp=drivesdk"

# Extract file ID from the link
file_id = excel_file_url.split("id=")[-1] if "id=" in excel_file_url else excel_file_url.split("/d/")[1].split("/")[0]

# Download the Excel file
excel_file = "1.xlsx"
gdown.download(f"https://drive.google.com/uc?id={file_id}", excel_file, quiet=False)

# Define the dataset directory
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)  # Create dataset directory if it doesn't exist

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file)

# Loop through each student entry in the DataFrame
for index, row in df.iterrows():
    roll_number = row['Roll Number']
    images = [
        row['Please Upload Your 5 Different Face Images (Not Full Body Image)For Training Purpose - Image 1'],
        row['Image 2'],
        row['Image 3'],
        row['Image 4'],
        row['Image 5']
    ]

    student_dir = os.path.join(dataset_dir, str(roll_number))
    if os.path.exists(student_dir):
        print(f"Folder for roll number {roll_number} already exists, skipping...")
        continue
    
    # Create a folder for the student based on their roll number
    os.makedirs(student_dir, exist_ok=True)
    
    # Loop through the images and download each if it's a valid URL
    for i, img_url in enumerate(images):
        if pd.notna(img_url):  # Check if the URL is not NaN
            try:
                # Extract the file ID from Google Drive link
                file_id = img_url.split("id=")[-1] if "id=" in img_url else img_url.split("/d/")[1].split("/")[0]
                output_path = os.path.join(student_dir, f"image_{i+1}.jpg")
                
                # Use gdown to download the file
                gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
                print(f"Downloaded image {i+1} for roll number {roll_number}")
                
            except Exception as e:
                print(f"Failed to download image {i+1} for roll number {roll_number}: {e}")

print("All images downloaded and organized.")
