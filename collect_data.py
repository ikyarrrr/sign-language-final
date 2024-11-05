# collect_data.py
import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# List of target words
words = ["hello", "thank you", "yes", "no", "goodbye", "please", "sorry", "help"]
dataset_size = 100  # Number of frames per word

# Function to find the first available camera index
def find_camera():
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    return -1

camera_index = find_camera()
if camera_index == -1:
    print("Error: No available video stream")
    exit()

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Error: Could not open video stream at index {camera_index}")
    exit()

for word in words:
    word_dir = os.path.join(DATA_DIR, word)
    if not os.path.exists(word_dir):
        os.makedirs(word_dir)

    print(f"Recording for word: {word}")
    input("Press Enter to start recording...")

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        cv2.putText(frame, f"Recording: {word} ({counter}/{dataset_size})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(word_dir, f'{counter}.jpg'), frame)

        counter += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
