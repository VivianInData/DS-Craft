import numpy as np
from sklearn.neighbors import NearestNeighbors

class RecommenderModel:
    def __init__(self):
        self.knn_model = None
        self.item_features = None
        self.item_to_index = None
        self.idx_to_item = None

    def prepare_data(self, item_features, item_ids):
        # Store the preprocessed features
        self.item_features = item_features
        
        # Handle NaN values in features
        self.item_features = np.nan_to_num(self.item_features, 0)
        
        # Create mappings
        self.item_to_index = {item: idx for idx, item in enumerate(item_ids)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_index.items()}

    def train(self):
        self.knn_model = NearestNeighbors(
            n_neighbors=20, 
            metric='cosine', 
            algorithm='brute', 
            n_jobs=-1
        )
        self.knn_model.fit(self.item_features)

    def get_recommendations(self, user_id, user_items, n_recommendations=10):
        recommendations = set()
        
        for item_id in user_items:
            if item_id in self.item_to_index:
                item_idx = self.item_to_index[item_id]
                _, indices = self.knn_model.kneighbors(
                    self.item_features[item_idx].reshape(1, -1)
                )
                similar_items = [self.idx_to_item[idx] for idx in indices[0]]
                recommendations.update(similar_items)

        # Remove items the user has already interacted with
        recommendations = recommendations - set(user_items)
        
        return list(recommendations)[:n_recommendations]