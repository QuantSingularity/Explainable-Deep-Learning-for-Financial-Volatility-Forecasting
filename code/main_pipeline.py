"""
Main Pipeline: Complete Workflow for Model Training, Evaluation, and Explanation
"""

import numpy as np
import tensorflow as tf
import os

# Set seeds
np.random.seed(123)
tf.random.set_seed(456)

# Import custom modules
from data_generator import generate_synthetic_dataset
from utils import load_and_prepare_data, train_val_test_split
from model import get_attention_weights
from train import train_model, plot_training_history
from eval import (
    evaluate_volatility_forecast,
    evaluate_var_backtest,
    compare_with_baselines,
    plot_var_backtest,
    generate_results_table,
)
from explain import run_full_explainability_pipeline


def main():
    """Execute complete pipeline."""

    print("\n" + "=" * 70)
    print("LSTM-ATTENTION-SHAP COMPLETE PIPELINE")
    print("=" * 70)

    # ==================================================================
    # STEP 1: DATA PREPARATION
    # ==================================================================
    print("\n[STEP 1/5] DATA PREPARATION")
    print("-" * 70)

    data_path = "../data/synthetic_data.csv"

    # Check if data exists, otherwise generate
    if not os.path.exists(data_path):
        print("Data not found. Generating synthetic dataset...")
        df = generate_synthetic_dataset(n_days=1827, start_date="2018-01-01")
        df.to_csv(data_path, index=False)

    # Load and split data
    df, feature_cols = load_and_prepare_data(data_path)
    data_dict = train_val_test_split(
        df,
        feature_cols,
        target_col="realized_volatility",
        train_end="2022-12-31",
        val_end="2023-06-30",
    )

    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols[:5]}...")

    # ==================================================================
    # STEP 2: MODEL TRAINING
    # ==================================================================
    print("\n[STEP 2/5] MODEL TRAINING")
    print("-" * 70)

    model_path = "../models/lstm_attention_model.h5"

    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "pinball_loss": lambda y_true, y_pred: tf.reduce_mean(
                    tf.maximum(0.01 * (y_true - y_pred), (0.01 - 1) * (y_true - y_pred))
                )
            },
        )
    else:
        print("Training new model...")
        model, history = train_model(
            data_dict, epochs=100, batch_size=64, save_path="../models"
        )
        plot_training_history(history, save_path="../figures")

    # ==================================================================
    # STEP 3: MODEL EVALUATION
    # ==================================================================
    print("\n[STEP 3/5] MODEL EVALUATION")
    print("-" * 70)

    # Volatility forecasting evaluation
    results, predictions_dict = evaluate_volatility_forecast(
        model, data_dict, data_dict["scalers"]["target"]
    )

    # VaR backtesting
    var_results = evaluate_var_backtest(predictions_dict, alpha=0.01)

    # Compare with baselines
    comparison_df = compare_with_baselines(predictions_dict)

    # Generate plots
    plot_var_backtest(var_results, predictions_dict, save_path="../figures")

    # Save tables
    generate_results_table(comparison_df, var_results, save_path="../paper")

    # ==================================================================
    # STEP 4: EXPLAINABILITY ANALYSIS
    # ==================================================================
    print("\n[STEP 4/5] EXPLAINABILITY ANALYSIS")
    print("-" * 70)

    # Extract attention weights
    print("Extracting attention weights...")
    attention_weights = get_attention_weights(model, data_dict["test"]["X"])

    # Run SHAP analysis
    explain_results = run_full_explainability_pipeline(
        model,
        data_dict["train"]["X"],
        data_dict["test"]["X"],
        data_dict["feature_names"],
        attention_weights=attention_weights,
    )

    # ==================================================================
    # STEP 5: SUMMARY
    # ==================================================================
    print("\n[STEP 5/5] PIPELINE SUMMARY")
    print("-" * 70)

    print("\n✓ Data prepared and split chronologically")
    print(f"✓ Model trained (RMSE: {results['RMSE']:.4f})")
    print(
        f"✓ VaR backtesting completed (Violation rate: {var_results['violation_rate']*100:.2f}%)"
    )
    print("✓ SHAP analysis completed")
    print(f"✓ Top driver: {explain_results['importance_df'].iloc[0]['Feature']}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("  Models: ../models/lstm_attention_model.h5")
    print("  Figures: ../figures/*.png")
    print("  Tables: ../paper/*.csv")
    print("  Reports: ../paper/*.txt")

    return {
        "model": model,
        "data": data_dict,
        "results": results,
        "var_results": var_results,
        "explain_results": explain_results,
    }


if __name__ == "__main__":
    pipeline_results = main()
    print("\nAll tasks completed. Ready for paper generation.")
