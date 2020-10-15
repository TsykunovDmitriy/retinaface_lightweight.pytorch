import numpy as np
import cv2

class KalmanFilter:
    '''A kalman filter for 1D or 2D points'''
    
    def __init__(self, input_dim=2, cov_process = 0.001, cov_measure=0.01, initial_state=None):
        '''Initialization the stablilizer, 1D for scaler, 2D for 1 point(x y)
        Input: initial_state --- a list of integers to initialize kalman filter
                            1 entry for 1D point, 2 entries for 2D points.
                input_dim --- the dimension of the points for kalman filter. 1 or 2.
                cov_process --- process covarience
                cov_measure --- measure covariance'''
        
        # Check: only iput dimension 1 or to allowed
        assert input_dim==1 or input_dim==2, "only 1D or 2D allowed"
        
        # Set up process and input dimensions
        self.state_num = 2*input_dim
        self.measure_num = input_dim
        
        # initiate filter from opencv
        # No control parameter for now
        self.filter = cv2.KalmanFilter(self.state_num, self.measure_num, 0)

        self.need_init = True
        if initial_state:
            self.init_state(initial_state)
        

        # initialize the measurements with 0s
        self.measurement = np.zeros((self.measure_num, 1), np.float32)
        
        # initialize the prediction results with 0s  
        self.prediction = np.zeros((self.state_num, 1), np.float32)
        
        # Kalman filter parameters setup for 1D
        if self.measure_num==1:
            self.filter.transitionMatrix = np.array([[1, 1],
                                                     [0, 1]], np.float32)
            self.filter.measurementMatrix = np.array([[1,1]], np.float32)
            self.filter.processNoiseCov = np.array([[1, 0], 
                                                    [0, 1]], np.float32)*cov_process
            self.filter.measurementNoiseCov = np.array([[1]], np.float32)*cov_measure
            
            
        # Kalman filter parameters setup for 2D
        if self.measure_num == 2:
            self.filter.transitionMatrix = np.array([[1,0,1,0],
                                                     [0,1,0,1],
                                                     [0,0,1,0],
                                                     [0,0,0,1]], np.float32)
            self.filter.measurementMatrix = np.array([[1,0,0,0],
                                                      [0,1,0,0]], np.float32)
            self.filter.processNoiseCov = np.array([[1,0,0,0],
                                                    [0,1,0,0],
                                                    [0,0,1,0],
                                                    [0,0,0,1]], np.float32)*cov_process
            self.filter.measurementNoiseCov = np.array([[1,0],
                                                        [0,1]], np.float32)*cov_measure

    def init_state(self, initial_state):
        if self.measure_num==1:
            self.filter.statePost = np.array([
                    [initial_state[0]],
                    [0]
                ], np.float32)
        if self.measure_num==2:
            self.filter.statePost = np.array([
                    [initial_state[0]],
                    [initial_state[1]],
                    [0],
                    [0]
                ], np.float32)
        self.need_init = False

    def update(self, measurement):
        '''update the kalman filter, containing both prediction by previous results, and the 
        correction with new measurements. Results are stored in the self.state.
        Input: measurement --- the new measurement to update kalman filter'''
        # make prediction based on previous results with kalman filter

        if self.need_init:
            self.init_state(measurement)

        self.prediction = self.filter.predict()
        
        # Get new measurements
        if self.measure_num == 1:
            self.measurement = np.array([[np.float32(measurement)]])
        else:
            self.measurement = np.array([[np.float32(measurement[0])],
                                         [np.float32(measurement[1])]])
        
        # correct according to measurement
        self.filter.correct(self.measurement)
        
        # update the state value
        self.state = self.filter.statePost
        
    def predict(self):
        # make prediction based on previous results with kalman filter
        self.prediction = self.filter.predict()
        self.state = self.prediction
        # return the prediction results
        return self.state
    
    def correct(self, measurement):
        # Correct the prediciton with new measurements
        if self.measure_num == 1:
            self.measurement = np.array([[np.float32(measurement)]])
        else:
            self.measurement = np.array([[np.float32(measurement[0])],
                                         [np.float32(measurement[1])]])
        
        # correct according to measurement
        self.filter.correct(self.measurement)
        # update the state value
        self.state = self.filter.statePost
        
        # return corrected results
        return self.state
    
    
    def get_results(self):
        # get the state of the kalman filter
        if(self.state_num==2):
            return self.state[0]
        if(self.state_num==4):
            return [self.state[0][0], self.state[1][0]]