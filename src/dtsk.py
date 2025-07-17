import numpy as np
from kddcup import load_kddcup99
import pickle
import logging

class D_TSK_FC:
    def __init__(self, DP, K_dp_list, alpha, C):
        self.DP = DP
        self.K_dp_list = K_dp_list
        self.alpha = alpha
        self.C = C

        self.centers = np.array([0, 0.25, 0.5, 0.75, 1])
        self.sigmas = None
        self.U_matrices = {}   #  Υ random 
        self.RC_matrices = {}  #  RC random 
        self.Z_matrices = {}   #  Z random 
        self.betas = {}        #  β weights

    def _gaussian_mf(self, x, center, sigma):
        return np.exp(-((x - center)**2) / (2 * sigma**2))

    def _calculate_h(self, x_sample, dp):
        N, d = x_sample.shape
        K_dp = self.K_dp_list[dp-1]
        
        gamma = self.U_matrices[dp]  # Shape: (d, K_dp)
        rc = self.RC_matrices[dp]    # Shape: (d, 5, K_dp)
        
        u_values = self._gaussian_mf(x_sample[:, :, np.newaxis], self.centers[np.newaxis, np.newaxis, :], self.sigmas[np.newaxis, np.newaxis, :])
        
        term = rc[np.newaxis, :, :, :] * u_values[:, :, :, np.newaxis]
        v_jl = 1 - np.prod(1 - term, axis=2)
        v_jl_selected = np.where(gamma[np.newaxis, :, :] == 1, v_jl, 1)
        
        H_dp = np.prod(v_jl_selected, axis=1)
        return H_dp

    def fit(self, X, T):
        """
        - X (np.ndarray): (N, d)
        - T (np.ndarray): one-hot (N, m)
        """
        N, d = X.shape
        m = T.shape[1]

        self.sigmas = np.random.rand(5) 
        # randomly initialize sigma values for Gaussian membership functions

        X_original = X.copy()
        Y_prev = None

        for dp in range(1, self.DP + 1):
            print(f"--- Training Layer {dp} ---")
            
            if dp == 1:
                X_dp = X_original
                # for the first layer, use the original input
            else:
                X_dp = X_original + self.alpha * (Y_prev @ self.Z_matrices[dp-1])
                # calculate the perturbed input for the current layer

            K_dp = self.K_dp_list[dp-1]
            self.U_matrices[dp] = np.random.randint(0, 2, size=(d, K_dp))
            self.RC_matrices[dp] = np.random.randint(0, 2, size=(d, 5, K_dp))
            
            print("Calculating hidden layer H...")
            H_dp = np.zeros((N, K_dp))
            for i in range(N):
                H_dp[i, :] = self._calculate_h(X_dp[i], dp)
                
            I = np.identity(K_dp)               # identity matrix for regularization
            H_T_H = H_dp.T @ H_dp + 1e-8 * I    # regularization term
            term1 = np.linalg.inv((1/self.C) * I + H_T_H)
            term2 = H_dp.T @ T
            self.betas[dp] = term1 @ term2
            
            if dp < self.DP:
                Y_dp = H_dp @ self.betas[dp]
                Y_prev = Y_dp
                
                self.Z_matrices[dp] = np.random.rand(m, d)
        
        print("\nTraining complete.")

    def predict(self, X_new):
        print("\n--- Starting Prediction ---")
        N_new = X_new.shape[0]
        predictions = []

        for i in range(N_new):
            z_original = X_new[i]
            z_dp = z_original.copy()

            for dp in range(1, self.DP + 1):
                h_dp = self._calculate_h(z_dp.flatten(), dp).reshape(1, -1)
                
                y_dp = h_dp @ self.betas[dp]
                
                if dp < self.DP:
                    z_dp = z_original + self.alpha * (y_dp @ self.Z_matrices[dp])
            predicted_class = np.argmax(y_dp)
            predictions.append(predicted_class)
            
        return np.array(predictions)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"D-TSK-FC model saved to {filepath}")

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"D-TSK-FC model loaded from {filepath}")
        return model

if __name__ == '__main__':
    kdd_file_path = 'kddcup.data'

    try:
        X_train, T_train, X_test, y_test_true, num_classes = load_kddcup99(kdd_file_path)
        
        print(f"X_train shape: {X_train.shape}")
        print(f"T_train shape: {T_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test_true shape: {y_test_true.shape}")

        model_params = {
            'DP': 3,
            'K_dp_list': [80, 50, 30],
            'alpha': 0.03,
            'C': 1000
        }
        
        dtsk_fc = D_TSK_FC(**model_params)
        dtsk_fc.fit(X_train, T_train)
        
        y_pred = dtsk_fc.predict(X_test)
        
        print("\n--- Results ---")
        accuracy = np.mean(y_pred == y_test_true) * 100
        print(f"Accuracy: {accuracy:.2f}%")

    except FileNotFoundError:
        print(f"\nERROR: Data file not found at '{kdd_file_path}'.")
