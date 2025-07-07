#!/usr/bin/env python3

"""
Kubeflow pipeline for VSP eyeglass frame style prediction
"""

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["pandas", "numpy", "scikit-learn", "python-dotenv", "boto3"],
)
def load_and_prepare_data(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    data_filename: str,
    processed_data: Output[Dataset],
    label_encoder: Output[Model],
    labels: Output[Dataset],
    descriptions: Output[Dataset],
):
    """Load and prepare data for the embedding pipeline"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    import pickle
    import boto3
    import os

    # Setup MinIO client
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False,
    )

    # Download data from MinIO
    local_data_path = "/tmp/input_data.csv"
    s3_client.download_file(bucket_name, f"data/{data_filename}", local_data_path)

    # Load the synthetic eyeglass frame data
    df = pd.read_csv(local_data_path)

    print(f"Loaded {len(df)} records from MinIO bucket {bucket_name}")
    print(f"Columns: {list(df.columns)}")

    # Prepare labels
    label_encoder_obj = LabelEncoder()
    y = label_encoder_obj.fit_transform(df["Frame_Style"])
    class_names = label_encoder_obj.classes_

    print(f"Number of frame styles: {len(class_names)}")
    print(f"Frame styles: {list(class_names)}")

    # Save processed data and encoder
    df.to_csv(processed_data.path, index=False)

    with open(label_encoder.path, "wb") as f:
        pickle.dump(label_encoder_obj, f)

    with open(labels.path, "wb") as f:
        pickle.dump(y, f)

    # Save descriptions for embedding generation
    descriptions_list = df["Frame_Description"].tolist()
    with open(descriptions.path, "wb") as f:
        pickle.dump(descriptions_list, f)

    print("Data preparation complete")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["numpy", "python-dotenv", "requests"],
)
def generate_embeddings(
    descriptions: Input[Dataset],
    endpoint: str,
    embedding_model: str,
    api_key: str,
    embeddings: Output[Dataset],
):
    """Generate embeddings using vLLM BGE-Large model"""
    import pickle
    import numpy as np
    import requests
    import json

    # Load descriptions
    with open(descriptions.path, "rb") as f:
        descriptions_list = pickle.load(f)

    print(f"Generating embeddings for {len(descriptions_list)} descriptions...")

    # Create embeddings via API call
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    embeddings_list = []

    for desc in descriptions_list:
        payload = {"model": embedding_model, "input": desc}

        response = requests.post(
            f"{endpoint}/v1/embeddings", headers=headers, json=payload, verify=False
        )

        if response.status_code == 200:
            result = response.json()
            embedding = result["data"][0]["embedding"]
            embeddings_list.append(embedding)
        else:
            print(f"Error generating embedding: {response.status_code}")
            raise Exception(f"Failed to generate embedding for: {desc[:50]}...")

    if embeddings_list:
        X = np.array(embeddings_list)
        print(f"Generated embeddings with shape: {X.shape}")

        # Save embeddings
        with open(embeddings.path, "wb") as f:
            pickle.dump(X, f)

        print("Embeddings saved")
    else:
        raise Exception("Failed to generate embeddings")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["numpy", "scikit-learn", "joblib"],
)
def train_model(
    embeddings: Input[Dataset],
    labels: Input[Dataset],
    label_encoder: Input[Model],
    trained_model: Output[Model],
    test_results: Output[Dataset],
    metrics: Output[Metrics],
):
    """Train KNN classifier on embeddings"""
    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    import json

    # Load embeddings and labels
    with open(embeddings.path, "rb") as f:
        X = pickle.load(f)

    with open(labels.path, "rb") as f:
        y = pickle.load(f)

    with open(label_encoder.path, "rb") as f:
        label_encoder_obj = pickle.load(f)

    class_names = label_encoder_obj.classes_

    print(f"Dataset info:")
    print(f"   - Total samples: {len(X)}")
    print(f"   - Embedding dimension: {X.shape[1]}")
    print(f"   - Number of frame styles: {len(class_names)}")

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data: {len(X_train)} train, {len(X_test)} test")

    # Train KNN classifier
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    print(f"Training KNN classifier with k={k}...")
    knn.fit(X_train, y_train)
    print("KNN training complete")

    # Evaluate model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f"   - Accuracy: {accuracy:.1%}")

    # Save model
    joblib.dump(knn, trained_model.path)

    # Save test data for evaluation
    test_data = {
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "class_names": class_names.tolist(),
    }

    with open(test_results.path, "wb") as f:
        pickle.dump(test_data, f)

    # Log metrics
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("training_samples", len(X_train))
    metrics.log_metric("test_samples", len(X_test))
    metrics.log_metric("embedding_dimension", X.shape[1])
    metrics.log_metric("num_classes", len(class_names))

    print("Model training and evaluation complete")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["matplotlib", "seaborn", "scikit-learn", "numpy", "boto3"],
)
def evaluate_model(
    test_results: Input[Dataset],
    label_encoder: Input[Model],
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    confusion_matrix_plot: Output[Dataset],
):
    """Evaluate model and create visualizations"""
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    import boto3
    import os

    # Load test results and label encoder
    with open(test_results.path, "rb") as f:
        test_data = pickle.load(f)

    with open(label_encoder.path, "rb") as f:
        label_encoder_obj = pickle.load(f)

    X_test = np.array(test_data["X_test"])
    y_test = np.array(test_data["y_test"])
    y_pred = np.array(test_data["y_pred"])
    class_names = test_data["class_names"]

    print(f"Evaluation on {len(X_test)} test samples")
    print(f"Frame styles: {class_names}")

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - VSP Eyeglass Frame Style Prediction")
    plt.xlabel("Predicted Frame Style")
    plt.ylabel("Actual Frame Style")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plot_path = "/tmp/confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Upload to MinIO
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False,
    )

    s3_client.upload_file(
        plot_path, bucket_name, "results/confusion_matrix.png"
    )

    # Save plot path
    with open(confusion_matrix_plot.path, "w") as f:
        f.write(plot_path)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Model evaluation complete")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "requests",
        "boto3",
    ],
)
def predict_on_unseen_data(
    trained_model: Input[Model],
    label_encoder: Input[Model],
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    unseen_data_filename: str,
    predictions_output_filename: str,
    endpoint: str,
    embedding_model: str,
    api_key: str,
):
    """Make predictions on unseen eyeglass frame data"""
    import pickle
    import numpy as np
    import pandas as pd
    import requests
    import json
    import joblib
    import boto3

    # Setup MinIO client
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False,
    )

    # Download unseen data from MinIO
    local_data_path = "/tmp/unseen_data.csv"
    s3_client.download_file(bucket_name, f"data/{unseen_data_filename}", local_data_path)

    # Load unseen data
    unseen_df = pd.read_csv(local_data_path)
    print(f"Loaded {len(unseen_df)} unseen frame descriptions")

    # Load trained model and label encoder
    knn = joblib.load(trained_model.path)
    with open(label_encoder.path, "rb") as f:
        label_encoder_obj = pickle.load(f)

    class_names = label_encoder_obj.classes_

    # Generate embeddings for unseen data
    descriptions = unseen_df["Frame_Description"].tolist()
    print(f"Generating embeddings for {len(descriptions)} unseen descriptions...")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    embeddings_list = []

    for desc in descriptions:
        payload = {"model": embedding_model, "input": desc}

        response = requests.post(
            f"{endpoint}/v1/embeddings", headers=headers, json=payload, verify=False
        )

        if response.status_code == 200:
            result = response.json()
            embedding = result["data"][0]["embedding"]
            embeddings_list.append(embedding)
        else:
            print(f"Error generating embedding: {response.status_code}")
            raise Exception(f"Failed to generate embedding for: {desc[:50]}...")

    if embeddings_list:
        X_unseen = np.array(embeddings_list)
        print(f"Generated embeddings with shape: {X_unseen.shape}")

        # Make predictions
        y_pred_unseen = knn.predict(X_unseen)
        y_pred_proba = knn.predict_proba(X_unseen)

        # Convert predictions back to frame style names
        predicted_styles = label_encoder_obj.inverse_transform(y_pred_unseen)

        # Create results DataFrame
        results_df = pd.DataFrame({
            "Frame_Description": descriptions,
            "Predicted_Frame_Style": predicted_styles,
            "Confidence": np.max(y_pred_proba, axis=1)
        })

        # Add probability scores for each class
        for i, class_name in enumerate(class_names):
            results_df[f"Prob_{class_name}"] = y_pred_proba[:, i]

        print(f"Predictions complete for {len(results_df)} unseen frames")
        print(f"Predicted frame styles: {results_df['Predicted_Frame_Style'].value_counts().to_dict()}")

        # Save results locally
        local_results_path = "/tmp/predictions_output.csv"
        results_df.to_csv(local_results_path, index=False)

        # Upload to MinIO
        s3_client.upload_file(
            local_results_path, bucket_name, f"results/{predictions_output_filename}"
        )

        print(f"Results saved to MinIO: results/{predictions_output_filename}")
    else:
        raise Exception("Failed to generate embeddings for unseen data")


@pipeline(
    name="vsp-eyeglass-frame-pipeline",
    description="VSP eyeglass frame style prediction using BGE-Large embeddings",
)
def vsp_eyeglass_frame_pipeline(
    minio_endpoint: str = "",
    minio_access_key: str = "",
    minio_secret_key: str = "",
    bucket_name: str = "pipeline",
    data_filename: str = "synthetic_eyeglass_frames_1k.csv",
    endpoint: str = "",
    embedding_model: str = "",
    api_key: str = "",
    unseen_data_filename: str = "unseen_eyeglass_frames.csv",
    predictions_output_filename: str = "predictions_output.csv",
):
    """VSP Eyeglass Frame Style Prediction Pipeline"""
    
    # Step 1: Load and prepare data
    data_prep = load_and_prepare_data(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        data_filename=data_filename,
    )

    # Step 2: Generate embeddings
    embeddings = generate_embeddings(
        descriptions=data_prep.outputs["descriptions"],
        endpoint=endpoint,
        embedding_model=embedding_model,
        api_key=api_key,
    ).after(data_prep)

    # Step 3: Train model
    model_training = train_model(
        embeddings=embeddings.outputs["embeddings"],
        labels=data_prep.outputs["labels"],
        label_encoder=data_prep.outputs["label_encoder"],
    ).after(embeddings)

    # Step 4: Evaluate model
    evaluation = evaluate_model(
        test_results=model_training.outputs["test_results"],
        label_encoder=data_prep.outputs["label_encoder"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
    ).after(model_training)

    # Step 5: Predict on unseen data
    predictions = predict_on_unseen_data(
        trained_model=model_training.outputs["trained_model"],
        label_encoder=data_prep.outputs["label_encoder"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        unseen_data_filename=unseen_data_filename,
        predictions_output_filename=predictions_output_filename,
        endpoint=endpoint,
        embedding_model=embedding_model,
        api_key=api_key,
    ).after(model_training)


if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        vsp_eyeglass_frame_pipeline,
        "vsp_eyeglass_frame_pipeline.yaml",
    )
