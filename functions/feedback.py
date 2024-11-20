from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Abstract base class for the Strategy
class SegmentationStrategy(ABC):
    @abstractmethod
    def fit(self, vectors):
        """Pre-compute any necessary data for the strategy."""
        pass
    
    @abstractmethod
    def assign_segment(self, vector):
        """Assign a new vector to a segment based on the strategy."""
        pass

    @abstractmethod
    def move_closer(self, vector, closest, alpha):
        """Get a vector that will move the input closest to center."""
        pass

# Strategy 1: KMeans Clustering
class KMeansSegmentation(SegmentationStrategy):
    def __init__(self, num_sections=11):
        self.num_sections = num_sections
        self.kmeans = None

    def fit(self, vectors):
        """Train KMeans on the sample vectors to create num_sections clusters."""
        self.kmeans = KMeans(n_clusters=self.num_sections, random_state=42, n_init=10, max_iter=300)
        self.kmeans.fit(vectors)

    def move_closer(self, vector, closest, alpha):
        # vector = np.array(vector).reshape(1, -1)
        center = self.kmeans.cluster_centers_[closest]
        # print(vector)
        # print(center)
        adj = alpha * (center - vector)
        # print(adj)
        return adj

    def assign_segment(self, vector):
        """Assign the vector to the nearest KMeans cluster center."""
        vector = np.array(vector).reshape(1, -1)
        closest, _ = pairwise_distances_argmin_min(vector, self.kmeans.cluster_centers_)
        return closest[0]

# Strategy 2: Hyperplane Division
class HyperplaneSegmentation(SegmentationStrategy):
    def __init__(self, num_sections=11):
        self.num_sections = num_sections
        self.thresholds = None

    def fit(self, vectors):
        """Compute thresholds for dividing the vector space evenly along each dimension."""
        # Calculate the min and max for each dimension, then evenly divide across num_sections
        mins = np.min(vectors, axis=0)
        maxs = np.max(vectors, axis=0)
        self.thresholds = np.linspace(mins, maxs, self.num_sections + 1, axis=0)

    def assign_segment(self, vector):
        """Assign a vector to a segment based on hyperplane thresholds."""
        # Determine which segment the vector falls into by comparing against thresholds
        segment = 0
        for i, val in enumerate(vector):
            segment += np.searchsorted(self.thresholds[:, i], val, side='right') - 1
        # Use modulo operation to return a segment label between 0 and num_sections - 1
        return segment % self.num_sections
    
    def move_closer(self, vector, closest, alpha):
        vector = np.array(vector).reshape(1, -1)
        center = self.kmeans.cluster_centers_[closest]
        adj = alpha * (center - vector)
        return adj

# Context Class for Segmentation
class VectorSegmenter:
    def __init__(self, strategy: SegmentationStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: SegmentationStrategy):
        """Set a new strategy."""
        self.strategy = strategy
    
    def fit(self, vectors):
        """Fit the strategy with the given vectors (if required)."""
        self.strategy.fit(vectors)

    def move_closer(self, vector, closest, alpha):
        """Get a vector that will move the input closest to center."""
        return self.strategy.move_closer(vector, closest, alpha)
    
    def assign_segment(self, vector):
        """Assign a segment to the vector using the current strategy."""
        return self.strategy.assign_segment(vector)

