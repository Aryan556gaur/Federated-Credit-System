import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# --- Config ---
NUM_CLIENTS = 5
NUM_ROUNDS = 5
BATCH_SIZE = 32
CLIENT_LEARNING_RATE = 0.0001  # safer value to avoid NaN
SEED = 42

# Set seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --- Load and preprocess data ---
@st.cache
def load_and_preprocess():
    data = pd.read_csv(r"data\train.csv")
    if 'ID' in data.columns:
        data.drop(['ID'], axis=1, inplace=True)
    if 'Customer_ID' in data.columns:
        data.drop(['Customer_ID'], axis=1, inplace=True)

    # Features and target
    feature_columns = [col for col in data.columns if col != 'Credit_Score']
    X = data[feature_columns]
    y = data['Credit_Score']

    # Label encoding for categorical features
    cat_cols = X.select_dtypes(include='object').columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Scale numeric
    scaler = StandardScaler()
    num_cols = X.select_dtypes(exclude='object').columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    num_classes = len(target_encoder.classes_)

    return X.values.astype(np.float32), y.astype(np.int32), num_classes


def create_dataset(x, y, num_classes, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=len(x))
    ds = ds.batch(batch_size)
    ds = ds.map(lambda a, b: (a, tf.one_hot(b, depth=num_classes)))
    return ds


def partition_data(X, y):
    client_data = {}
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    splits = np.array_split(indices, NUM_CLIENTS)

    for i in range(NUM_CLIENTS):
        idx = splits[i]
        client_data[f'client_{i+1}'] = (X[idx], y[idx])
    return client_data


def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CLIENT_LEARNING_RATE, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model


def train_one_epoch(model, client_data, num_classes):
    X, y = client_data
    dataset = create_dataset(X, y, num_classes)
    history = model.fit(dataset, epochs=1, verbose=0)
    return history.history, model.get_weights()


def aggregate(weights_list):
    avg_weights = [np.zeros_like(w) for w in weights_list[0]]
    for weights in weights_list:
        for i in range(len(weights)):
            avg_weights[i] += weights[i] / len(weights_list)
    return avg_weights


# --- Streamlit UI ---
st.title("📊 Federated Learning Demo - Credit Score Prediction")

# Load data
with st.spinner("Loading and preprocessing data..."):
    X, y, num_classes = load_and_preprocess()
    client_data = partition_data(X, y)
    input_shape = (X.shape[1],)

st.success("Data loaded successfully!")
st.write(f"🔹 Number of Clients: {NUM_CLIENTS}")
st.write(f"🔹 Classes: {num_classes}")
st.write(f"🔹 Features: {input_shape[0]}")

# Global model
global_model = build_model(input_shape, num_classes)
initial_weights = global_model.get_weights()

round_metrics = []

if st.button("🚀 Start Federated Training"):
    for round_num in range(NUM_ROUNDS):
        st.write(f"### 🔄 Round {round_num+1}")

        client_weights = []
        round_losses = []
        round_accuracies = []

        for client_id in client_data.keys():
            client_model = build_model(input_shape, num_classes)
            client_model.set_weights(global_model.get_weights())
            hist, weights = train_one_epoch(client_model, client_data[client_id], num_classes)
            client_weights.append(weights)
            round_losses.append(hist['loss'][0])
            round_accuracies.append(hist['categorical_accuracy'][0])

        avg_weights = aggregate(client_weights)
        global_model.set_weights(avg_weights)

        avg_loss = np.mean(round_losses)
        avg_acc = np.mean(round_accuracies)
        round_metrics.append((avg_loss, avg_acc))

        st.write(f"🔹 Avg Loss: {avg_loss:.4f} | Avg Accuracy: {avg_acc:.4f}")

    st.success("✅ Training complete!")

    # Evaluate on all clients
    st.write("### 🧪 Evaluating on Clients")
    for cid, (cx, cy) in client_data.items():
        ds = create_dataset(cx, cy, num_classes)
        loss, acc = global_model.evaluate(ds, verbose=0)
        st.write(f"{cid} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Save weights
    global_model.save_weights("federated_model_weights.h5")
    st.success("💾 Global model weights saved!")

    # Plot metrics
    import matplotlib.pyplot as plt

    rounds = list(range(1, NUM_ROUNDS+1))
    losses = [m[0] for m in round_metrics]
    accs = [m[1] for m in round_metrics]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(rounds, losses, marker='o')
    ax[0].set_title("Loss Over Rounds")
    ax[1].plot(rounds, accs, marker='o', color='green')
    ax[1].set_title("Accuracy Over Rounds")
    st.pyplot(fig)
