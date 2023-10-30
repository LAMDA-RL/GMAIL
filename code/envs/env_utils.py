import joblib


def save_path(samples, filename):
    joblib.dump(samples, filename, compress=3)
    