import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Pastikan path ini sesuai dengan file automate Anda
from automate_Nalendra import load_and_preprocess_data

def train_model():
    print("============= proses training =============")

    # 1. Load data
    # Menggunakan path dinamis agar aman dijalankan dari mana saja
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "titanic_clean.csv") # Atau sesuaikan path csv Anda
    
    # Jika file csv ada di folder yang berbeda (misal di folder parent), sesuaikan path-nya
    # Contoh jika csv di folder MLProject dan script di folder yang sama, kode di atas sudah benar.
    if not os.path.exists(csv_path):
         # Fallback jika dijalankan dari root repository
         csv_path = "Titanic_raw.csv"

    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_path)

    # 2. Aktifkan MLflow Autolog
    # Ini akan otomatis melacak parameter, metrik training, dan menyimpan model
    mlflow.sklearn.autolog()

    # Set Experiment
    # mlflow.set_experiment("Titanic_Survival_Prediction")

    with mlflow.start_run():
        
        # Parameter Model
        n_estimators = 100
        random_state = 42
        
        # pipeline setup
        pipeline = Pipeline(steps=[
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            ))
        ])

        # Training
        print("============= training model =============")
        # Saat .fit() dipanggil, autolog otomatis mencatat:
        # - Parameter (n_estimators, dll)
        # - Model Artifact (model.pkl)
        # - Training metrics (jika ada)
        pipeline.fit(X_train, y_train)

        # Evaluation (Test Set)
        # Autolog biasanya hanya mencatat metrik data training. 
        # Metrik data test tetap kita hitung manual agar tercatat di MLflow.
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Test Accuracy: {acc}")

        # Kita log metrik test secara manual agar hasil evaluasi validasi muncul di dashboard
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        # Tidak perlu mlflow.sklearn.log_model(...) lagi karena sudah di-handle autolog
        print("Model dan parameter berhasil disimpan otomatis oleh Autolog.")

if __name__ == "__main__":
    train_model()
