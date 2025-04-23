import os
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Configuration ---
DATA_PATH = 'data/train.csv'
NUM_CLIENTS = 5
BATCH_SIZE = 32
NUM_ROUNDS = 10
CLIENT_LEARNING_RATE = 0.001
SEED = 42

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# --- Data Loading with Robust Error Handling ---
def load_data():
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
            
        train_data = pd.read_csv(DATA_PATH)
        
        if len(train_data) == 0:
            raise ValueError("Training data is empty")
            
        return train_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.warning("Generating synthetic data for demonstration")
        return pd.DataFrame({
            'Age': np.random.randint(18, 70, 1000),
            'Income': np.random.normal(5000, 2000, 1000),
            'Debt': np.random.normal(1000, 500, 1000),
            'Credit_Score': np.random.choice(['Good', 'Standard', 'Poor'], 1000)
        })

# --- Preprocessing ---
def preprocess_data(train_data):
    try:
        feature_columns = [col for col in train_data.columns if col not in ['ID', 'Customer_ID', 'Credit_Score']]
        X = train_data[feature_columns]
        y = train_data['Credit_Score']
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(exclude=['object']).columns
        
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Scale numerical features
        scaler = StandardScaler()
        if len(numerical_cols) > 0:
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # Encode target
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        
        return X.values.astype(np.float32), y.astype(np.int32), len(target_encoder.classes_)
        
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        raise

# --- Model Definition ---
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CLIENT_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Federated Training ---
def federated_training(X, y, num_classes):
    try:
        # Verify dimensions
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
        
        # Split data among clients
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, NUM_CLIENTS)
        
        # Create and train global model
        global_model = create_model((X.shape[1],), num_classes)
        
        for round_num in range(NUM_ROUNDS):
            client_weights = []
            
            for i, indices in enumerate(client_indices):
                client_X, client_y = X[indices], y[indices]
                
                # Create and train client model
                client_model = create_model((X.shape[1],), num_classes)
                client_model.set_weights(global_model.get_weights())
                
                dataset = tf.data.Dataset.from_tensor_slices((
                    client_X,
                    tf.one_hot(client_y, num_classes)
                )).batch(BATCH_SIZE)
                
                client_model.fit(dataset, epochs=1, verbose=0)
                client_weights.append(client_model.get_weights())
            
            # Aggregate weights
            avg_weights = [np.mean([w[i] for w in client_weights], axis=0) 
                          for i in range(len(client_weights[0]))]
            global_model.set_weights(avg_weights)
            
        return global_model
        
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        raise

# --- Streamlit App ---
def main():
    st.title("🏦 Federated Credit Scoring System")
    
    if st.button("Train Model"):
        with st.spinner("Loading data..."):
            train_data = load_data()
            st.write(f"Loaded {len(train_data)} samples")
            
        with st.spinner("Preprocessing data..."):
            X, y, num_classes = preprocess_data(train_data)
            st.write(f"Input shape: {X.shape}, Target shape: {y.shape}")
            
        with st.spinner("Training federated model..."):
            model = federated_training(X, y, num_classes)
            st.success("Training completed!")
            
            # Save model
            os.makedirs('models', exist_ok=True)
            model.save('models/federated_model.h5')
            st.write("Model saved to 'models/federated_model.h5'")

if __name__ == "__main__":
    main()