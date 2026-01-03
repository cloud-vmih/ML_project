import numpy as np

class KNNRegressorCustom:
    """K-Nearest Neighbors implement từ đầu"""
    
    def __init__(self, k=5, weights='uniform', metric='euclidean'):
        self.k = k
        self.weights = weights
        self.metric = metric
        self.X_train = None
        self.y_train = None
        
    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y).flatten()
        return self
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for x_test in X:
            # Tính distances
            distances = []
            for x_train in self.X_train:
                dist = self._distance(x_test, x_train)
                distances.append(dist)
            
            distances = np.array(distances)
            
            # Tìm k nearest neighbors
            indices = np.argsort(distances)[:self.k]
            k_distances = distances[indices]
            k_labels = self.y_train[indices]
            
            # Tính prediction
            if self.weights == 'uniform':
                pred = np.mean(k_labels)
            elif self.weights == 'distance':
                weights = 1 / (k_distances + 1e-8)
                pred = np.average(k_labels, weights=weights)
            else:
                raise ValueError(f"Unknown weight: {self.weights}")
            
            predictions.append(pred)
        
        return np.array(predictions)