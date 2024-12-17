import csv
import numpy as np
import time
import pywt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score


class DataProcessor:
    def __init__(self, anno_path, dat_list_path):
        self.anno_path = anno_path  # Path to CSV annotations
        self.dat_list_path = dat_list_path  # Path to list of data files
        self.X = []  # Feature matrix
        self.y = []  # Target array
        self.features = []  # Extracted features

    def process_data(self):
        """
        Loads and processes raw data from files.
        """
        print("Loading files:", self.anno_path, "and", self.dat_list_path)
        start_time = time.time()

        num_files = sum(1 for _ in open(self.dat_list_path))
        self.X = np.empty((num_files, 8, 2200), dtype=np.int16)
        self.y = []

        with open(self.anno_path, 'r') as csv_file, open(self.dat_list_path, 'r') as list_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the header row in CSV

            for i, (row, file_path) in enumerate(zip(csv_reader, list_file)):
                file_path = file_path.strip()
                self.X[i] = np.fromfile(file_path, dtype=np.int16).reshape((8, -1))
                values = [int(value) for value in row]
                self.y.append(values)

        print(f"Data loading completed in {time.time() - start_time:.2f} seconds")

    def extract_features(self):
        """
        Extracts statistical, frequency, and wavelet-based features from the loaded data.
        """
        sampling_rate = 300  # Sampling rate (unused, but could be needed later)
        x = np.array(self.X)
        n_files, n_channels, n_samples = x.shape

        print("Starting feature extraction...")
        for i in range(n_files):
            feature_list = []
            for n in range(n_channels):
                channel_data = x[i][n]

                # Statistical features
                feature_list.append(np.mean(channel_data))
                feature_list.append(np.std(channel_data))
                feature_list.append(np.var(channel_data))
                feature_list.append(np.mean(np.diff(np.sign(channel_data))))  # Difference in sign
                feature_list.append(np.max(channel_data) - np.min(channel_data))  # Amplitude range

                # Frequency features using FFT
                fft_result = np.fft.fft(channel_data)
                power_spectrum = np.abs(fft_result) ** 2 / n_samples
                dominant_freq = np.argmax(power_spectrum)
                feature_list.append(dominant_freq)

                # Top 3 frequencies
                top_frequencies = np.argsort(power_spectrum)[::-1][:3]
                feature_list.extend(top_frequencies)

                # Wavelet transform features
                coeffs = pywt.dwt(channel_data, 'db1')
                wavelet_energy = np.sum(np.abs(coeffs) ** 2)
                feature_list.append(wavelet_energy)

            self.features.append(feature_list)
        print("Feature extraction completed.")


def train_and_evaluate(train_paths, dev_paths, k_neighbors):
    """
    Trains and evaluates a k-NN classifier on the training and development sets.

    Args:
        train_paths (tuple): (anno_path, dat_list_path) for training data.
        dev_paths (tuple): (anno_path, dat_list_path) for development data.
        k_neighbors (int): Number of neighbors for k-NN.
    """
    # Process training data
    print("\n### Processing Training Data ###")
    dp_train = DataProcessor(*train_paths)
    dp_train.process_data()
    dp_train.extract_features()
    X_train = np.array(dp_train.features)
    y_train = np.array(dp_train.y)

    # Process development data
    print("\n### Processing Development Data ###")
    dp_dev = DataProcessor(*dev_paths)
    dp_dev.process_data()
    dp_dev.extract_features()
    X_dev = np.array(dp_dev.features)
    y_dev = np.array(dp_dev.y)

    # Train k-NN Classifier
    print("\n### Training and Evaluating k-NN Classifier ###")
    classifier = KNN(n_neighbors=k_neighbors)
    classifier.fit(X_train, y_train)

    # Evaluate on Training Set
    predictions_train = classifier.predict(X_train)
    acc_train = accuracy_score(y_train, predictions_train)
    print(f"Training Accuracy: {acc_train:.4f}")

    # Evaluate on Development Set
    predictions_dev = classifier.predict(X_dev)
    acc_dev = accuracy_score(y_dev, predictions_dev)
    print(f"Development Accuracy: {acc_dev:.4f}")


if __name__ == "__main__":
    # Define file paths
    train_healthy_paths = ("./set_15/lists/data_train_healthy.csv", "set_15/lists/data_train_healthy.list")
    train_unhealthy_paths = ("./set_15/lists/data_train_unhealthy.csv", "set_15/lists/data_train_unhealthy.list")

    dev_healthy_paths = ("./set_15/lists/data_dev_healthy.csv", "set_15/lists/data_dev_healthy.list")
    dev_unhealthy_paths = ("./set_15/lists/data_dev_unhealthy.csv", "set_15/lists/data_dev_unhealthy.list")

    # Combine training and development paths
    train_paths = (
        [train_healthy_paths[0], train_unhealthy_paths[0]],
        [train_healthy_paths[1], train_unhealthy_paths[1]],
    )
    dev_paths = (
        [dev_healthy_paths[0], dev_unhealthy_paths[0]],
        [dev_healthy_paths[1], dev_unhealthy_paths[1]],
    )

    # Train and evaluate the model
    k_neighbors = 6  # Choose the k value for k-NN
    train_and_evaluate(train_healthy_paths, dev_healthy_paths, k_neighbors)
