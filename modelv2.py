from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# Reading dataset
dataset = pd.read_csv("pincode-dataset.csv", header=0, encoding='unicode_escape')
x = dataset['Pincode']
y = dataset['District']

# Splitting into training and validation dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)

# Creating model instance
model = DecisionTreeClassifier()

# Training model
model.fit(x_train.to_frame(), y_train)

# Evaluating the model
y_pred = model.predict(x_test.to_frame())
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")

# Saving the model locally
with open('iModelv2.pkl', 'wb') as file:
    pickle.dump(model, file)
