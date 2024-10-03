import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load Data 
data = pd.read_csv('heart_failure.csv')
print(data.info())

# Print distribution of death_event column (classification labels)
print('Classes and number of values in the dataset', Counter(data['death_event']))

# Split data into label and feature sets
y = data['death_event']
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]
 
# One-hot encode categorical features
x = pd.get_dummies(x)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)

# Standardize numeric features
column_transformer = ColumnTransformer([('numeric', StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])

X_train = column_transformer.fit_transform(X_train)
X_test = column_transformer.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train.astype(str))
y_test = label_encoder.transform(y_test.astype(str))

# Transform encoded labels into binary vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build neural network model
model = Sequential()
model.add(InputLayer(input_shape = (X_train.shape[1], )))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

# Compile model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Train and evaluate model
model.fit(X_train, y_train, epochs = 100, batch_size = 16, verbose = 1)
loss, accuracy = model.evaluate(X_train, y_train, verbose = 0)
print("Loss:", loss, "Accuracy:", accuracy)

# Classification report
y_estimate = model.predict(X_test, verbose = 0)
y_estimate = np.argmax(y_estimate, axis = 1)
y_true = np.argmax(y_test, axis = 1)
print(classification_report(y_true, y_estimate))
