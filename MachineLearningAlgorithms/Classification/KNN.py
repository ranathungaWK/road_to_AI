import numpy as np

def euclidean_distance(point1 , point2):
    return np.sqrt(np.sum((point1 - point2)**2))

class KNN :
    
    def __init__(self , k=3):
        self.k = k

    def fit(self , x , y):
        self.X_train = x
        self.Y_train = y

    def predict(self , x):
        predictions = [self._predict(x_test)for x_test in x]
        return np.array(predictions)
    
    def _predict(self, x_test):
        distances = [euclidean_distance(x_test , x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        return np.argmax(np.bincount(k_nearest_labels))
    
    def accuracy(self , y_true , y_pred):
        accuracy = np.sum(y_true == y_pred)/len(y_true)
        return accuracy
    
    
