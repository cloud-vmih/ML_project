import numpy as np

class DecisionTreeCustom:
    """Decision Tree từ đầu (dùng cho Random Forest)"""
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)
    
    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_mse = float('inf')
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                
                mse_left = self._mse(y[left_mask])
                mse_right = self._mse(y[right_mask])
                
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                weighted_mse = (n_left * mse_left + n_right * mse_right) / (n_left + n_right)
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'value': np.mean(y)}
        
        # Find best split
        feature_idx, threshold = self._best_split(X, y)
        
        if feature_idx is None:
            return {'value': np.mean(y)}
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_one(self, x, node):
        if 'value' in node:
            return node['value']
        
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
    
    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(x, self.tree) for x in X])


class RandomForestCustom:
    """Random Forest implement từ đầu"""
    
    def __init__(self, n_estimators=100, max_depth=10, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        
        n_samples, n_features = X.shape
        
        # Xác định số features cho mỗi split
        if self.max_features is None:
            max_features = int(np.sqrt(n_features))
        else:
            max_features = self.max_features
        
        # Tạo các trees
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Feature sampling
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            X_bootstrap = X_bootstrap[:, feature_indices]
            
            # Tạo và train tree
            tree = DecisionTreeCustom(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            
            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices
            })
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            X_subset = X[:, feature_indices]
            pred = tree.predict(X_subset)
            predictions.append(pred)
        
        # Average predictions
        return np.mean(predictions, axis=0)