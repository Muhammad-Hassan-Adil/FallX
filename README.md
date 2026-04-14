# FallX: Machine Learning Fall Detection Model

FallX is an intelligent, low-cost fall detection and protection system designed for vulnerable populations, including the elderly and motorcyclists. This repository contains the Python-based Machine Learning pipeline responsible for real-time fall classification.

The model processes 9-axis motion tracking data (accelerometer, gyroscope, and magnetometer) to reliably differentiate between actual falls (forward, backward, lateral) and routine activities of daily living (ADLs). 

## Model Architecture
The core fall detection engine utilizes a **Convolutional Neural Network (CNN)** optimized for deployment on resource-constrained microcontrollers (ESP32). 

* **Input:** Raw time-series data from MPU9250 sensors (Chest and Waist).
* **Architecture:** Sequential CNN layers with Batch Normalization, MaxPooling, Dropout, and GlobalMaxPooling, ending in a Softmax activation dense layer.
* **Performance:** * **Accuracy:** 94%
  * **Precision:** 1.00
  * **Recall:** 0.46
* **Why CNN?** Outperformed Random Forest and SVM in balancing high detection accuracy with the low false-positive rates necessary for physical airbag deployment.

## Dataset & Preprocessing
Trained and validated using the **UMAFall Dataset**.

1. **Data Cleaning:** Removed invalid rows, filtered specific sensor IDs, and split data into distinct accelerometer/gyroscope/magnetometer readings.
2. **Feature Engineering:** Captured sudden spikes in angular velocity (Z-axis for lateral falls) and rapid acceleration shifts (X/Y/Z axes for forward/backward falls).
3. **Scaling:** Features standardized using `scikit-learn`'s `StandardScaler`.
4. **Labeling:** Binary classification mapping `backwardFall`, `forwardFall`, and `lateralFall` to `1` and routine movements to `0`.
