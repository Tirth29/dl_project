import face_recognition
import cv2
import pickle
import os
import pandas as pd
from datetime import datetime

class FaceRecognizer:
    def __init__(self, encoding_file="encodings.pkl", labels_file="labels.pkl", results_file="recognition_results.xlsx"):
        self.encoding_file = encoding_file
        self.labels_file = labels_file
        self.results_file = results_file
        self.load_encodings()

    def load_encodings(self):
        # Load known face encodings and labels from files
        with open(self.encoding_file, "rb") as f:
            self.known_encodings = pickle.load(f)
        with open(self.labels_file, "rb") as f:
            self.known_labels = pickle.load(f)

    def recognize_faces(self, test_img_path, output_dir="test_results"):
        # Load and convert the test image
        test_img = face_recognition.load_image_file(test_img_path)
        test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings in the test image
        face_locations = face_recognition.face_locations(test_img)
        face_encodings = face_recognition.face_encodings(test_img, face_locations)

        recognized_roll_numbers = []  # To store recognized roll numbers

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.45)
            label = "Unknown"
            if True in matches:
                match_index = matches.index(True)
                label = self.known_labels[match_index]
                recognized_roll_numbers.append(label)

            # Draw rectangle and label on the image
            cv2.rectangle(test_img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(test_img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the annotated image
        os.makedirs(output_dir, exist_ok=True)
        annotated_test_img_path = os.path.join(output_dir, "annotated_test_image.jpg")
        cv2.imwrite(annotated_test_img_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))

        # Save results to Excel
        self.save_results_to_excel(recognized_roll_numbers, test_img_path)

        print(f"Annotated test image saved in '{output_dir}' directory.")

    def save_results_to_excel(self, recognized_roll_numbers, test_img_path):
        # Prepare data for saving
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "DateTime": [now],
            "Recognized Roll Numbers": [", ".join(recognized_roll_numbers)],
            "Test Image Path": [test_img_path]
        }
        df = pd.DataFrame(data)

        # Append to Excel file, create a new one if it doesn't exist
        if os.path.exists(self.results_file):
            with pd.ExcelWriter(self.results_file, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            df.to_excel(self.results_file, index=False)

        print(f"Recognition results saved to '{self.results_file}'.")

# Usage
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    test_img_path = input("Enter the path to a test image: ")
    recognizer.recognize_faces(test_img_path)
