from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from typing import List, Dict

class MetricsCalculator:
    """Calculates evaluation metrics for model benchmarks."""
    
    @staticmethod
    def calculate_metrics(results: List[Dict]) -> Dict:
        """
        Expects a list of dicts with:
        - true_label: int (0 for normal, 1 for fault)
        - pred_score: float (percentage or raw)
        - pred_label: int (0 or 1)
        """
        y_true = [getattr(r, "true_label", 1 if r["True Label"] == "fault" else 0) for r in results]
        y_scores = [r["Score %"] / 100.0 for r in results]
        y_pred = [1 if r["Pred Label"] == "fail" else 0 for r in results]
        
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0)
        }
        
        # AUROC requires scores from both classes to be present
        if len(set(y_true)) > 1:
            metrics["AUROC"] = roc_auc_score(y_true, y_scores)
        else:
            metrics["AUROC"] = 0.0
            
        return metrics
