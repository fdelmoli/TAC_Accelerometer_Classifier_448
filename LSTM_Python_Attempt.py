import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error

def generate_data_chunks(data, seq_length, batch_size):
    for i in range(0, len(data) - seq_length - 1, batch_size):
        X_batch = []
        y_batch = []
        for j in range(batch_size):
            if i + j + seq_length >= len(data):
                break  # Stop iteration if index exceeds data size
            X = data[i + j:i + j + seq_length, :-1]  # Exclude last column (TAC) for input
            y = data[i + j + seq_length, -1]         # Last column (TAC) for output
            X_batch.append(X)
            y_batch.append(y)
        if len(X_batch) < batch_size:  # Ensure batch size consistency
            break  # Stop iteration if remaining data is insufficient for a full batch
        yield np.array(X_batch), np.array(y_batch)


def main():
    merged = pd.read_parquet('merged_data.parquet')
    merged = merged.drop(columns=['time', 'pid'])

    # Generating sample accelerometer and TAC data
    # DataFrame with columns 'x', 'y', 'z', and 'tac'
    data = merged.values

    # Normalizing data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Splitting data into training and testing sets
    seq_length = 50 # arbitrary number
    X_train, X_test, y_train, y_test = train_test_split(scaled_data[:, :-1], scaled_data[:, -1], test_size=0.2, random_state=42)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    batch_size = 32  # Adjust as needed
    train_data_generator = generate_data_chunks(X_train, seq_length, batch_size)
    test_data_generator = generate_data_chunks(X_test, seq_length, batch_size)

    # Building LSTM model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dense(units=1))  # Output dimension is 1 for TAC prediction
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training the model
    steps_per_epoch_train = (len(X_train) - seq_length - 1) // batch_size
    steps_per_epoch_test = (len(X_test) - seq_length - 1) // batch_size
    model.fit(train_data_generator, epochs=1, steps_per_epoch=steps_per_epoch_train,
                        validation_data=test_data_generator, validation_steps=steps_per_epoch_test, verbose=1)

    # # Plotting training and validation loss
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig('training_validation_loss.png') 

    # Making predictions
    test_sequences = np.array([X_test[i:i+seq_length] for i in range(len(X_test) - seq_length)])
    predicted_tac = model.predict(test_sequences)

    # Rescale data back to original range
    predicted_tac = scaler.inverse_transform(predicted_tac)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predicted_tac)
    print("Mean Squared Error:", mse)

    # Plotting predictions
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual TAC')
    plt.plot(predicted_tac, label='Predicted TAC')
    plt.title('TAC Prediction')
    plt.xlabel('Samples')
    plt.ylabel('TAC')
    plt.legend()
    plt.savefig('tac_prediction.png')

if __name__ == "__main__":
    main()
