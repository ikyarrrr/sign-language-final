# train_classifier.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']
max_length = max(len(x) for x in data)
padded_data = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in data])

x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, stratify=labels)

param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
print(f"Test set accuracy: {best_model.score(x_test, y_test) * 100:.2f}%")

with open('model.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)
print("Model saved successfully.")
