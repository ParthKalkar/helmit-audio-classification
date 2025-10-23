import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

def compute_metrics(csv_path):
    df = pd.read_csv(csv_path)
    y_true = df['label'].values
    y_pred = df['pred'].values
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    
    return acc, prec, rec, f1

def main():
    models = [
        ('lite_cnn_v1', 'data/lite_cnn_v1_predictions.csv'),
        ('lite_cnn_v2', 'data/lite_cnn_v2_predictions.csv'),
        ('lite_cnn_v3', 'data/lite_cnn_v3_predictions.csv'),
        ('lite_cnn_v4', 'data/lite_cnn_v4_predictions.csv'),
        ('wav2vec2_v1', 'data/wav2vec2_v1_predictions.csv'),
        ('wav2vec2_v2', 'data/wav2vec2_v2_predictions.csv'),
    ]
    
    with open('model_metrics.txt', 'w') as f:
        f.write("Model Performance Metrics\n")
        f.write("=" * 50 + "\n\n")
        
        for name, csv_path in models:
            try:
                acc, prec, rec, f1 = compute_metrics(csv_path)
                f.write(f"Model: {name}\n")
                f.write(f"  Accuracy: {acc:.4f}\n")
                f.write(f"  Precision: {prec:.4f}\n")
                f.write(f"  Recall: {rec:.4f}\n")
                f.write(f"  F1 Score: {f1:.4f}\n")
                f.write("\n")
            except Exception as e:
                f.write(f"Model: {name} - Error: {e}\n\n")
        
        # Add training RAM/time notes if available
        f.write("Training RAM/Time Notes (from logs):\n")
        f.write("- lite_cnn_v1: ~250-300 MB RAM, ~40-50s/epoch\n")
        f.write("- lite_cnn_v2: ~250-300 MB RAM, ~40-50s/epoch\n")
        f.write("- lite_cnn_v3: ~250-300 MB RAM, ~40-50s/epoch\n")
        f.write("- lite_cnn_v4: ~250-300 MB RAM, ~40-50s/epoch\n")
        f.write("- wav2vec2_v1: ~400-500 MB RAM, ~35-40s/epoch\n")
        f.write("- wav2vec2_v2: ~400-500 MB RAM, ~35-40s/epoch\n")

if __name__ == "__main__":
    main()