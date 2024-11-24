import face_recognition
import os
import pickle

class FaceTrainer:
    def __init__(self, dataset_dir="dataset", encoding_file="encodings.pkl", labels_file="labels.pkl"):
        self.dataset_dir = dataset_dir
        self.encoding_file = encoding_file
        self.labels_file = labels_file
        self.known_encodings = []
        self.known_labels = []

    def load_existing_data(self):
        """Load existing encodings and labels if available."""
        if os.path.exists(self.encoding_file):
            with open(self.encoding_file, "rb") as f:
                self.known_encodings = pickle.load(f)
        if os.path.exists(self.labels_file):
            with open(self.labels_file, "rb") as f:
                self.known_labels = pickle.load(f)

    def train_faces(self):
        """Train new faces by adding encodings for new images."""
        self.load_existing_data()

        # Track processed images as (admission_no, img_name) pairs
        processed_images = set(zip(self.known_labels, range(len(self.known_labels))))  # Use index to avoid ambiguity
        new_encodings = []
        new_labels = []

        for admission_no in os.listdir(self.dataset_dir):
            person_dir = os.path.join(self.dataset_dir, admission_no)
            if os.path.isdir(person_dir):
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)

                    # Skip if this image has already been processed
                    if (admission_no, img_name) in processed_images:
                        continue

                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        new_encodings.append(encodings[0])
                        new_labels.append(admission_no)
                        print(f"New encoding added for {admission_no}: {img_name}")

        # Update and save encodings and labels
        self.known_encodings.extend(new_encodings)
        self.known_labels.extend(new_labels)

        with open(self.encoding_file, "wb") as f:
            pickle.dump(self.known_encodings, f)
        with open(self.labels_file, "wb") as f:
            pickle.dump(self.known_labels, f)

        print(f"Training complete. {len(new_encodings)} new encodings added.")

# Usage
if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.train_faces()
