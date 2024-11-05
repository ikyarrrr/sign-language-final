# sign_language_app.py
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

labels_dict = {i: word for i, word in enumerate(["hello", "thank you", "yes", "no", "goodbye", "please", "sorry", "help", "love", "friend"])}

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        self.sentence = ""
        self.cap = cv2.VideoCapture(0)
        self.last_pred_time = 0
        self.pred_delay = 1.5

        self.panel = tk.Label(root)
        self.panel.pack()
        self.pred_label = tk.Label(root, text="Sentence:", font=("Courier", 20))
        self.pred_label.pack(pady=10)

        self.process_frame()

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(1000, self.process_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        current_time = time.time()

        if results.multi_hand_landmarks and current_time - self.last_pred_time > self.pred_delay:
            self.last_pred_time = current_time
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    data_aux.extend([landmark.x, landmark.y])
            while len(data_aux) < 84:
                data_aux.extend([0, 0])
            prediction = model.predict([np.array(data_aux)])[0]
            self.sentence += " " + prediction
            self.pred_label.config(text=f"Sentence: {self.sentence}")

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)
        self.root.after(10, self.process_frame)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
