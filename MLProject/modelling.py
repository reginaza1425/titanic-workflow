import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from automate_Nalendra import load_and_preprocess_data


def train_model():

    print("============= proses training =============")

    # 1. Load data dari automate
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        "Titanic_raw.csv"
    )

    # 2. MLflow experiment
    mlflow.set_experiment("Titanic_Survival_Prediction")

    with mlflow.start_run():

        # Parameters
        n_estimators = 100
        random_state = 42

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)

        # Pipeline (optional tapi rapi)
        pipeline = Pipeline(steps=[
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            ))
        ])

        # Training
        print("============= training model =============")
        pipeline.fit(X_train, y_train)

        # Evaluation
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", acc)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save model
        mlflow.sklearn.log_model(pipeline, "model_titanic")

        print("Model dan metrics berhasil disimpan di MLflow")


if __name__ == "__main__":
    train_model()
