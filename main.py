from src.data.containers import RSSI
from src.data.partitions import TRAIN, TEST
from src.data.loader import load_dataset

from src.models.kNN import kNNLocalizer
from src.preprocessing.layers.kalman_filter import KalmanFilterLayer

import numpy as np

path = 'data\dataset1'

# Load dataset with desired data.
data = load_dataset(path, dtypes=[RSSI])

rssi_train = data[TRAIN][RSSI]
rssi_test  = data[TEST][RSSI]

# Initialize kNN model.
knn_localizer = kNNLocalizer(
    k=33,
    layers=[KalmanFilterLayer(Q=0.04, R=4.0)]
)

# Fit and predict
pred_y = knn_localizer.fit(rssi_train.X, rssi_train.y).predict(rssi_test.X)
true_y = rssi_test.y

# Calculate stats
errors = np.linalg.norm(pred_y - true_y, axis=1)

stats = {
    "mean": float(np.mean(errors)),
    "median": float(np.median(errors)),
    "p90": float(np.percentile(errors, 90)),
    "min": float(np.min(errors)),
    "max": float(np.max(errors)),
    "rmse": float(np.sqrt(np.mean(errors**2))),
}

print(stats)



