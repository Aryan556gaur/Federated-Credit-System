import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_and_preprocess_data(train_path, test_path):
    """Load and preprocess the data."""
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Extract features and target
    feature_columns = [col for col in train_data.columns if col not in ['ID', 'Customer_ID', 'Credit_Score']]
    X = train_data[feature_columns]
    y = train_data['Credit_Score']
    
    # Preprocess categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(exclude=['object']).columns
    
    # Create preprocessing pipelines
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    # Convert target to numeric
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    num_classes = len(target_encoder.classes_)
    
    # Convert to numpy arrays
    X = X.values.astype(np.float32)
    y = y.astype(np.int32)
    
    # Save preprocessing objects
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(target_encoder, 'models/target_encoder.pkl')
    joblib.dump(feature_columns, 'models/feature_columns.pkl')
    
    return X, y, num_classes, scaler, label_encoders, target_encoder, feature_columns