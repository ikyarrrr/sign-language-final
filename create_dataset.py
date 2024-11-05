# create_dataset.py
import os
import pickle
import mediapipe as mp
import cv2
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

DATA_DIR = './data'
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

for word in os.listdir(DATA_DIR):
    word_dir = os.path.join(DATA_DIR, word)
    if os.path.isdir(word_dir):
        print(f"Processing word: {word}")
        for img_path in os.listdir(word_dir):
            data_aux = []
            x_, y_ = [], []
            img = cv2.imread(os.path.join(word_dir, img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))
                data.append(data_aux)
                labels.append(word)

if data and labels:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Dataset created successfully.")
else:
    print("No data found.")
