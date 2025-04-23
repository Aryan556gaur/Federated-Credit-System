import numpy as np
import tensorflow as tf
from models.federated_model import create_model
from models.preprocessing import load_and_preprocess_data
import joblib

# Configuration
NUM_CLIENTS = 5
BATCH_SIZE = 32
NUM_ROUNDS = 8
CLIENT_LEARNING_RATE = 0.001
SEED = 42

def create_client_data(X, y, num_clients=NUM_CLIENTS):
    """Partition data among clients."""
    client_data = {}
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    client_partitions = np.array_split(indices, num_clients)
    
    for i in range(num_clients):
        client_indices = client_partitions[i]
        client_X = X[client_indices]
        client_y = y[client_indices]
        client_data[f'client_{i}'] = (client_X, client_y)
    
    return client_data

def create_dataset(x, y, batch_size=BATCH_SIZE):
    """Create TensorFlow dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, depth=3)))
    return dataset

def train_client_model(client_model, client_data):
    """Train model on client data."""
    x, y = client_data
    dataset = create_dataset(x, y)
    history = client_model.fit(dataset, epochs=1, verbose=0)
    return history.history, client_model.get_weights()

def aggregate_weights(weights_list):
    """Federated averaging of client weights."""
    avg_weights = [np.zeros_like(w) for w in weights_list[0]]
    for weights in weights_list:
        for i, w in enumerate(weights):
            avg_weights[i] += w / len(weights_list)
    return avg_weights

def federated_training():
    """Main federated training loop."""
    # Load and preprocess data
    X, y, num_classes, _, _, _, _ = load_and_preprocess_data(
        'data/train.csv', 'data/test.csv'
    )
    
    # Create client data
    client_data = create_client_data(X, y)
    
    # Create global model
    input_shape = (X.shape[1],)
    global_model = create_model(input_shape, num_classes, CLIENT_LEARNING_RATE)
    
    # Training loop
    print("Starting Federated Training...")
    for round_num in range(NUM_ROUNDS):
        print(f"\nRound {round_num + 1}/{NUM_ROUNDS}")
        
        client_weights = []
        client_metrics = []
        
        # Client training phase
        for client_id, data in client_data.items():
            client_model = create_model(input_shape, num_classes, CLIENT_LEARNING_RATE)
            client_model.set_weights(global_model.get_weights())
            
            metrics, weights = train_client_model(client_model, data)
            client_weights.append(weights)
            client_metrics.append(metrics)
        
        # Aggregate metrics
        round_loss = np.mean([m['loss'][0] for m in client_metrics])
        round_accuracy = np.mean([m['categorical_accuracy'][0] for m in client_metrics])
        print(f"Average client metrics - Loss: {round_loss:.4f}, Accuracy: {round_accuracy:.4f}")
        
        # Server aggregation
        global_weights = aggregate_weights(client_weights)
        global_model.set_weights(global_weights)
    
    # Save model
    global_model.save_weights('models/federated_credit_model_weights.weights.h5')
    print("\nTraining complete. Model weights saved.")

if __name__ == "__main__":
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    federated_training()