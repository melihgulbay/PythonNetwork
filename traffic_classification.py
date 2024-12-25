import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class TrafficClassifier:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.traffic_types = [
            'web', 'streaming', 'gaming', 'voip', 'file_transfer', 
            'database', 'email', 'hard_to_classify'
        ]
        
        self.edge_traffic_history = defaultdict(lambda: defaultdict(int))
        self.trained = False
        
        # Try to load pre-trained model
        if os.path.exists('traffic_model.joblib'):
            try:
                self.classifier = joblib.load('traffic_model.joblib')
                self.trained = True
            except:
                pass

    def _extract_features(self, packet):
        """Extract features from a packet for classification"""
        return np.array([
            packet.size,  # Packet size
            packet.priority,  # Priority level
            len(packet.path),  # Path length
            packet.latency,  # Latency
            1 if packet.packet_type == "control" else 0,  # Is control packet
            packet.creation_time  # Timestamp
        ]).reshape(1, -1)

    def _generate_synthetic_data(self):
        """Generate synthetic data for initial training"""
        X = []
        y = []
        
        # Generate synthetic samples for each traffic type
        for _ in range(1000):  # 1000 samples per type
            for traffic_type in self.traffic_types:
                if traffic_type == 'web':
                    features = [
                        np.random.normal(500, 200),  # Small-medium packets
                        np.random.randint(1, 3),     # Medium priority
                        np.random.randint(1, 5),     # Short paths
                        np.random.normal(5, 2),      # Low latency
                        0,                           # Not control
                        np.random.randint(0, 1000)   # Random timestamp
                    ]
                elif traffic_type == 'streaming':
                    features = [
                        np.random.normal(1200, 300),  # Large packets
                        np.random.randint(2, 4),      # High priority
                        np.random.randint(1, 3),      # Short paths
                        np.random.normal(3, 1),       # Very low latency
                        0,                            # Not control
                        np.random.randint(0, 1000)
                    ]
                elif traffic_type == 'gaming':
                    features = [
                        np.random.normal(300, 100),   # Small packets
                        np.random.randint(2, 4),      # High priority
                        np.random.randint(1, 3),      # Short paths
                        np.random.normal(2, 1),       # Minimal latency
                        0,                            # Not control
                        np.random.randint(0, 1000)
                    ]
                elif traffic_type == 'voip':
                    features = [
                        np.random.normal(200, 50),    # Very small packets
                        3,                            # Highest priority
                        np.random.randint(1, 3),      # Short paths
                        np.random.normal(2, 1),       # Minimal latency
                        0,                            # Not control
                        np.random.randint(0, 1000)
                    ]
                elif traffic_type == 'file_transfer':
                    features = [
                        np.random.normal(1400, 100),  # Maximum size packets
                        np.random.randint(0, 2),      # Low priority
                        np.random.randint(1, 7),      # Variable paths
                        np.random.normal(10, 5),      # High latency
                        0,                            # Not control
                        np.random.randint(0, 1000)
                    ]
                elif traffic_type == 'database':
                    features = [
                        np.random.normal(800, 200),   # Medium-large packets
                        np.random.randint(1, 3),      # Medium priority
                        np.random.randint(1, 4),      # Medium paths
                        np.random.normal(6, 2),       # Medium latency
                        0,                            # Not control
                        np.random.randint(0, 1000)
                    ]
                elif traffic_type == 'email':
                    features = [
                        np.random.normal(400, 200),   # Variable size
                        0,                            # Lowest priority
                        np.random.randint(1, 7),      # Variable paths
                        np.random.normal(15, 5),      # High latency
                        0,                            # Not control
                        np.random.randint(0, 1000)
                    ]
                else:  # hard_to_classify
                    features = [
                        np.random.normal(600, 400),   # Highly variable size
                        np.random.randint(0, 4),      # Random priority
                        np.random.randint(3, 10),     # Long paths
                        np.random.normal(20, 10),     # Very high latency
                        np.random.randint(0, 2),      # Maybe control
                        np.random.randint(0, 1000)
                    ]
                
                X.append(features)
                y.append(traffic_type)
        
        return np.array(X), np.array(y)

    def train(self):
        """Train the classifier with synthetic data"""
        X, y = self._generate_synthetic_data()
        self.classifier.fit(X, y)
        self.trained = True
        
        # Save the trained model
        joblib.dump(self.classifier, 'traffic_model.joblib')

    def classify_packet(self, packet):
        """Classify a single packet"""
        if not self.trained:
            self.train()
            
        features = self._extract_features(packet)
        return self.classifier.predict(features)[0]

    def update_edge_traffic(self, edge, packet):
        """Update traffic statistics for an edge"""
        traffic_type = self.classify_packet(packet)
        self.edge_traffic_history[edge][traffic_type] += 1

    def get_edge_traffic_stats(self, edge):
        """Get traffic statistics for an edge"""
        total = sum(self.edge_traffic_history[edge].values())
        if total == 0:
            return {}
            
        return {
            traffic_type: count / total 
            for traffic_type, count in self.edge_traffic_history[edge].items()
        }

    def reset_stats(self):
        """Reset traffic statistics"""
        self.edge_traffic_history.clear()