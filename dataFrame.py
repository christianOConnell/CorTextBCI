import csv
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

def readEEG(filename, dataCube):
    # dataCube should contain marker and timestamp data
    with open(filename, 'r', newline='') as eeg:
        temp2 = ['f', dataCube[-1][1] + 4]
        dataCube.append(temp2)
        temp = []
        curr = dataCube[0]
        nextBoi = dataCube[1]
        startTime = float(curr[1]) #curr time
        nextInd = 2

        sums = [0,0,0,0,0,0,0,0] # 8 0s
        count = 0 # for finding averages

        eegreader = csv.reader(eeg, delimiter=',')
        next(eegreader) # skip headers on first line
        for row in eeg:
            splitRow = row.strip().split(',')
            if curr[0] == 'l' or curr[0] == -1:
                curr[0] = -1  # -1 for left
            else:
                curr[0] = 1  # 1 for right

            if float(splitRow[8]) < float(nextBoi[1]):
                if float(splitRow[8]) > startTime + 0.25:
                    for i in range(len(splitRow) - 1):
                        sums[i] += float(splitRow[i])
                    count += 1
                    if count >= 9:
                        for i in range(len(sums)):
                            sums[i] = int(sums[i])//10
                            temp.append(sums[i])
                        dataCube[nextInd - 2].append(temp)
                        temp = []
                        sums = [0, 0, 0, 0, 0, 0, 0, 0]  # 8 0s
                        count = 0

            else:
                curr = nextBoi
                if nextInd >= len(dataCube):
                    dataCube[nextInd - 2].pop(1)
                    dataCube[nextInd - 2].append(temp)
                    dataCube.pop(-1)
                    break
                nextBoi = dataCube[nextInd]
                dataCube[nextInd - 2].pop(1)
                dataCube[nextInd - 2].append(temp)
                nextInd += 1
                temp = []
                startTime = float(curr[1])

    return dataCube

def readMarkers(filename):
    with open(filename, 'r', newline='') as markers:
        dataCube = []
        eegreader = csv.reader(markers, delimiter=',')
        next(eegreader)
        firstRow = next(eegreader)
        firstTime = float(firstRow[1])
        dataCube.append([firstRow[0], 0])
        for row in markers:
            splitRow = row.strip().split(',')
            arr = [splitRow[0], round(float(splitRow[1]) - firstTime)] # mark timestamps start at 0
            dataCube.append(arr)
    return dataCube

# Define the filenames for marker and EEG data
markerCSV = "test3.csv"
eegCSV = "eegData1.csv"

# Read marker and EEG data from the CSV files
theDataCube = readMarkers(markerCSV)
theDataCube = readEEG(eegCSV, theDataCube)
#print(theDataCube)

# Extract valid EEG samples and their corresponding labels
valid_samples = []
labels = []


import pandas as pd


# Create empty DataFrames for each label
dfs = {1: pd.DataFrame(), -1: pd.DataFrame()}

# Iterate through theDataCube to create individual DataFrames
for sublist in theDataCube:
    label = sublist[0]  # Extract the label directly from the first element of each sublist
    if isinstance(label, int):  # Ensure label is an integer
        temp_data = []
        for sample in sublist[1:]:
            if isinstance(sample, list) and len(sample) >= 3:  # Ensure sample is a list and has at least 3 elements
                # Flatten the EEG data into a 1D array
                flattened_sample = np.array(sample[1:]).flatten()
                temp_data.append(flattened_sample)  # Append flattened EEG data to temporary list
            
        # Convert the temporary list to a DataFrame
        if temp_data:
            temp_df = pd.DataFrame(temp_data)
            temp_df['y'] = label  # Add label column to the DataFrame
            dfs[label] = pd.concat([dfs[label], temp_df], ignore_index=True)  # Concatenate with existing DataFrame
        #print("temp df")
        #print(temp_df)
        #print("new big df")
        #print(dfs)

counts = data['y'].value_counts()

# Display the counts
print("Number of instances where y = 1:", counts[1])
print("Number of instances where y = -1:", counts[-1])


# Reshape X to have shape (number of samples, number of channels, number of time steps)

# Check the shape of X before and after reshaping
print("Shape of X before reshaping:", X.shape)
print("Total size of X before reshaping:", X.size)
X = X.reshape(-1, 107, 8, 1) # assumes that each sample has 107 rows and 8 channels

# Adjust y to match the number of samples in X
y = y[:X.shape[0]]

# Convert labels to binary (-1, 1) format
y_binary = np.where(y == 0, -1, y)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Splitting the data for label 1
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    dfs[1].drop('y', axis=1),  # Drop the label column to get the features (X)
    dfs[1]['y'],  # Use the label column as the target variable (y)
    test_size=0.2,  # 80% training data, 20% testing data
    random_state=42,  # Random seed for reproducibility
)

# Splitting the data for label -1
X_train_neg1, X_test_neg1, y_train_neg1, y_test_neg1 = train_test_split(
    dfs[-1].drop('y', axis=1),  # Drop the label column to get the features (X)
    dfs[-1]['y'],  # Use the label column as the target variable (y)
    test_size=0.2,  # 80% training data, 20% testing data
    random_state=42,  # Random seed for reproducibility
)

# Concatenating the training and testing sets
X_train = pd.concat([X_train_1, X_train_neg1], ignore_index=True)
X_test = pd.concat([X_test_1, X_test_neg1], ignore_index=True)
y_train = pd.concat([y_train_1, y_train_neg1], ignore_index=True)
y_test = pd.concat([y_test_1, y_test_neg1], ignore_index=True)

# Display the shape of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)
print(X_train)
# Define CNN model


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(107, 8, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 2)))  # Max-pooling only along the channel dimension
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Predict probabilities for test set
y_pred_prob = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred_binary = (y_pred_prob > 0.5).astype(int)

# Calculate accuracy before switching
accuracy_before = np.mean(y_pred_binary == y_test)

# Switch predicted labels
y_pred_binary_switched = 1 - y_pred_binary

# Calculate accuracy after switching
accuracy_after = np.mean(y_pred_binary_switched == y_test)

#print(f'Accuracy before switching: {accuracy_before}')
#print(f'Accuracy after switching: {accuracy_after}')

print("\nPredicted vs Actual:")
for i in range(len(y_pred_binary)):
    print(f"Sample {i}: Predicted = {y_pred_binary[i]}, Actual = {y_test[i]}")
