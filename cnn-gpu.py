import os
import csv
import time
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Reproducibility
np.random.seed(22)

# ----------------------------------------------------------
# Data Processor Class
# ----------------------------------------------------------
class DataProcessor:
    """
    A class to handle processing of annotation and data files for multi-output classification.
    """
    def __init__(self, anno_path="", dat_list_path=""):
        self.anno_path = anno_path
        self.dat_list_path = dat_list_path
        self.X = []  # Feature matrix
        self.y = [[] for _ in range(6)]  # Multi-target labels for 6 outputs

    def process_data(self):
        """
        Reads annotation CSV and corresponding data files.
        """
        print(f"Loading files: {self.anno_path} and {self.dat_list_path}")
        start_time = time.time()

        with open(self.anno_path, 'r') as csv_file, open(self.dat_list_path, 'r') as list_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip header
            for i, (row, file_path) in enumerate(zip(csv_reader, list_file)):
                file_path = file_path.strip()

                if i % 1000 == 0:
                    print(f"Processed {i} records")

                # Load data
                data = np.fromfile(file_path, dtype=np.int16).reshape((-1, 8))
                labels = [int(value) for value in row]

                # Append one-hot encoded labels for multi-output classification
                for j in range(6):
                    self.y[j].append(to_categorical(labels[j], num_classes=2))
                self.X.append(data)

        print(f"Data processing completed in {time.time() - start_time:.2f} seconds.")

    def get_data(self):
        """
        Returns processed data.
        """
        return np.array(self.X), [np.array(y_target) for y_target in self.y]


# ----------------------------------------------------------
# Model Creation Function
# ----------------------------------------------------------
def create_model(input_shape=(2200, 8)):
    """
    Creates and returns a CNN model for binary classification.
    """
    input_layer = Input(shape=input_shape)

    # CNN layers
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)

    # Pooling, Dropout, and Dense layers
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(2, activation='softmax')(x)  # Binary classification

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# ----------------------------------------------------------
# Model Evaluation Function
# ----------------------------------------------------------
def evaluate_model(trainX, trainy, testX, testy, epochs=10, batch_size=32):
    """
    Compiles, trains, and evaluates a model on the given dataset.
    """
    trainX, trainy = shuffle(trainX, trainy)
    testX, testy = shuffle(testX, testy)

    print("Creating model...")
    model = create_model(input_shape=(2200, 8))
    model.summary()

    # Compile model
    print("Compiling the model...")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    print("Training model...")
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1)

    return model


# ----------------------------------------------------------
# Main Function to Process Data and Train Models
# ----------------------------------------------------------
if __name__ == "__main__":
    # Define file paths for training and evaluation
    train_files = [
        ("./set_15/lists/data_train_healthy.csv", "./set_15/lists/data_train_healthy.list"),
        ("./set_15/lists/data_train_unhealthy.csv", "./set_15/lists/data_train_unhealthy.list"),
    ]
    eval_files = [
        ("./set_15/lists/data_dev_healthy.csv", "./set_15/lists/data_dev_healthy.list"),
        ("./set_15/lists/data_dev_unhealthy.csv", "./set_15/lists/data_dev_unhealthy.list"),
    ]

    # Process training data
    train_proc = DataProcessor()
    for csv_path, list_path in train_files:
        train_proc.anno_path, train_proc.dat_list_path = csv_path, list_path
        train_proc.process_data()
    X_train, y_train = train_proc.get_data()

    # Process evaluation data
    eval_proc = DataProcessor()
    for csv_path, list_path in eval_files:
        eval_proc.anno_path, eval_proc.dat_list_path = csv_path, list_path
        eval_proc.process_data()
    X_eval, y_eval = eval_proc.get_data()

    # Display data dimensions
    print(f"Training data shape: {X_train.shape}")
    print(f"Evaluation data shape: {X_eval.shape}")

    # Train and evaluate models for all 6 outputs
    models = []
    predictions = []
    for i in range(6):
        print(f"\nTraining model for output {i}...")
        model = evaluate_model(X_train, y_train[i], X_eval, y_eval[i])
        models.append(model)

        # Generate predictions
        preds = model.predict(X_eval, batch_size=32, verbose=1)
        predictions.append(np.argmax(preds, axis=1))

    # Display predictions for all outputs
    for i, preds in enumerate(predictions):
        print(f"Predictions for output {i}: {preds}")
