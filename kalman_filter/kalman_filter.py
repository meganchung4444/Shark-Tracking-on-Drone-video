import numpy as np

class Kalman2d_mot(object):
    def __init__(self, params, detection):
        
        #Get the parameters for Kalman filter 
        dt=params["dt"]
        u_x = u_y = params["u"]
        std_acc = params["std_acc"]
        std_meas_x = params["std_meas_x"] 
        std_meas_y = params["std_meas_y"]         
        
        # Define sampling time
        self.dt = dt

        # Define the  control input matrix with a size 2x1 array
        self.u = np.array([[u_x],
                           [u_y]]) # 2x1 array

        # Initialize the initial state with a size 4x1 array
        self.x = np.array([[0], 
                           [0], 
                           [0], 
                           [0]]) # 4x1 array
        
        self.x[:2] = detection
                
        # Define the State Transition Matrix A
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # 4x4 array        

        # Define the Control Input Matrix B
        self.B = np.array([[(self.dt**2)/2, 0],
                           [(self.dt**2)/2, 0],
                           [self.dt,0],
                           [0,self.dt]])   # 4x2 array
        
        # Define Measurement Mapping Matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])  # 2x4 array

        # Initial Process Noise Covariance
        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                           [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                           [(self.dt**3)/2, 0, self.dt**2, 0],
                           [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2  # 4x4 array

        # Initial Measurement Noise Covariance
        self.R = np.array([[std_meas_x**2, 0],
                           [0, std_meas_y**2]]) # 2x2 array

        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1]) *1000  # 4x4 array

    def predict(self):
        
        # predict state
        #x_k =Ax_(k-1) + Bu_(k-1)     Eq.(2.4)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(2.5)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q        

        return self.x

    def update(self, z):
        
        # Calculate the Kalman Gain (Eq 2.6)
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        # K = P * H'* inv(S)  ==> S = H*P*H'+R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # update state  (Eq 2.7)
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))        
        
        # Update error covariance matrix (Eq 2.8)
        I = np.eye(self.H.shape[1])        
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x