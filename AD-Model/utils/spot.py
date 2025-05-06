
import numpy as np

class SPOT:
    def __init__(self, q=1e-4):
        self.q = q
        self.initialized = False

    def fit(self, init_data):
        self.init_data = np.asarray(init_data)
        self.init_threshold = np.percentile(self.init_data, 100 * (1 - self.q))
        self.peaks = self.init_data[self.init_data > self.init_threshold]
        self.initialized = True
        return self

    def run(self, test_data):
        if not self.initialized:
            raise RuntimeError("SPOT model must be fitted first.")
        test_data = np.asarray(test_data)
        pred = (test_data > self.init_threshold).astype(int)
        return self.init_threshold, pred
