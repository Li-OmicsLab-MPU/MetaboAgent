# src/tools/analysis/model_building_tools.py
"""
Model Building Tools - AutoGluon-based Machine Learning (Optimized)

Key Improvements:
1. Removed ACS metric. Leaderboards now rank by the specified eval_metric (default: ROC_AUC).
2. Integrated SMOTE+NearMiss resampling strictly within the training fold to prevent data leakage.
3. Feature Stability: Explicitly avoids any internal feature selection/pruning. 
   The model trains on exactly the feature set provided by Stage 5.

Privacy-Preserving Design:
All tools accept file paths instead of DataFrames. Data is processed locally.
"""

import pandas as pd
import numpy as np
import json
import os
import time
import shutil
import uuid
from typing import Dict, Any, List, Optional, Tuple

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        f1_score,
        recall_score,
        matthews_corrcoef,
        precision_score,
        confusion_matrix
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import NearMiss
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


def analyze_data_for_modeling(data_path: str, target_column: str) -> str:
    """
    Analyze dataset characteristics to provide modeling recommendations.
    """
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return json.dumps({"success": False, "error": "Unsupported file format"})
        
        if target_column not in df.columns:
            return json.dumps({"success": False, "error": f"Target column '{target_column}' not found"})
        
        n_samples = len(df)
        n_features = len(df.columns) - 1  # Exclude target
        
        # Target distribution
        target_dist = df[target_column].value_counts().to_dict()
        class_counts = list(target_dist.values())
        n_classes = len(class_counts)
        
        if n_classes < 2:
            return json.dumps({"success": False, "error": "Target column needs at least 2 classes"})
        
        imbalance_ratio = max(class_counts) / min(class_counts)
        
        # Recommend appropriate metric based on problem type
        if n_classes == 2:
            recommended_metric = "roc_auc"
        else:
            # For multiclass, use balanced_accuracy
            recommended_metric = "balanced_accuracy"
        
        return json.dumps({
            "success": True,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "class_distribution": {str(k): int(v) for k, v in target_dist.items()},
            "imbalance_ratio": round(imbalance_ratio, 2),
            "recommended_metric": recommended_metric,
            "recommendation": "High imbalance detected (>2.0)" if imbalance_ratio > 2.0 else "Balanced"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": f"Analysis failed: {str(e)}"})


def _safe_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return value


def _align_external_holdout(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Align external holdout columns to the training feature set."""
    feature_cols = [c for c in train_df.columns if c != target_column]
    aligned = holdout_df.copy()
    missing_features = [c for c in feature_cols if c not in aligned.columns]

    # Fill missing standardized features with 0.0 so the model can still score
    # on an externally projected holdout. On a scaled matrix, 0.0 is the
    # neutral mean-centered fallback.
    for feature in missing_features:
        aligned[feature] = 0.0

    keep_cols = feature_cols + ([target_column] if target_column in aligned.columns else [])
    aligned = aligned[keep_cols]
    return aligned, missing_features


def _compute_test_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
) -> Dict[str, Any]:
    auc_score: Any = "N/A"
    try:
        unique_classes = pd.Series(y_true).nunique()
        if y_prob is not None and unique_classes == 2:
            auc_score = round(roc_auc_score(y_true, y_prob), 4)
    except Exception:
        auc_score = "N/A"

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": auc_score,
        "f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def _to_jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _to_jsonable_list(values: List[Any]) -> List[Any]:
    return [_to_jsonable_scalar(v) for v in values]


def _train_with_sklearn_fallback(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_column: str,
    eval_metric: str,
    split_info: str,
    resampling_info: str,
    test_data_path: Optional[str] = None,
    missing_holdout_features: Optional[List[str]] = None,
) -> str:
    """Fallback trainer when AutoGluon is unavailable but sklearn is installed."""
    if not SKLEARN_AVAILABLE:
        return json.dumps({
            "success": False,
            "error": "Neither AutoGluon nor sklearn is available for model training."
        })

    feature_cols = [c for c in train_data.columns if c != target_column]
    aligned_test, missing_features = _align_external_holdout(train_data, test_data, target_column)
    if missing_holdout_features is not None:
        missing_features = list(missing_holdout_features)

    X_train = train_data[feature_cols]
    y_train = train_data[target_column]
    X_test = aligned_test[feature_cols]
    y_test = aligned_test[target_column]

    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    n_splits = max(2, min(5, min_class_count)) if min_class_count >= 2 else 0
    if n_splits < 2:
        return json.dumps({
            "success": False,
            "error": "Not enough samples per class for fallback cross-validation."
        })

    scoring = eval_metric if eval_metric in {"accuracy", "f1", "roc_auc"} else "roc_auc"
    if scoring == "f1":
        scoring = "f1_weighted"

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_specs = [
        ("LogisticRegression", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=42),
        )),
        ("RandomForest", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
        ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
        ("SVC", make_pipeline(
            StandardScaler(),
            SVC(probability=True, class_weight="balanced", random_state=42),
        )),
    ]

    leaderboard_with_predictions: List[Dict[str, Any]] = []
    best_payload: Optional[Dict[str, Any]] = None

    for model_name, model in model_specs:
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            score_val = float(np.mean(cv_scores))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_prob = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                    y_prob = proba[:, 1]
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(X_test)
                if isinstance(decision, np.ndarray):
                    decision = 1 / (1 + np.exp(-decision))
                    y_prob = decision

            metrics = _compute_test_metrics(y_test, y_pred, y_prob)
            holdout_score = metrics.get(eval_metric) if eval_metric in metrics else metrics.get("roc_auc")
            prediction_list = _to_jsonable_list(y_prob.tolist()) if isinstance(y_prob, np.ndarray) else None

            entry = {
                "model": model_name,
                "score_val": round(score_val, 4),
                "score_holdout": _safe_scalar(holdout_score),
                "predictions": prediction_list,
            }
            leaderboard_with_predictions.append(entry)

            sort_score = score_val
            if best_payload is None or sort_score > best_payload["sort_score"]:
                best_payload = {
                    "sort_score": sort_score,
                    "model_name": model_name,
                    "metrics": metrics,
                    "holdout_score": _safe_scalar(holdout_score),
                    "score_val": round(score_val, 4),
                }
        except Exception as e:
            leaderboard_with_predictions.append({
                "model": model_name,
                "score_val": None,
                "score_holdout": None,
                "predictions": None,
                "error": str(e),
            })

    if best_payload is None:
        return json.dumps({
            "success": False,
            "error": "sklearn fallback could not train any model successfully."
        })

    leaderboard_with_predictions.sort(
        key=lambda x: (-1e9 if x.get("score_val") is None else -float(x["score_val"]))
    )

    return json.dumps({
        "success": True,
        "best_model": best_payload["model_name"],
        "best_model_selection_basis": "train_cross_validation_fallback",
        "best_model_validation_score": best_payload["score_val"],
        "best_model_holdout_score": best_payload["holdout_score"],
        "primary_metric": eval_metric,
        "test_set_metrics": best_payload["metrics"],
        "leaderboard_top5": leaderboard_with_predictions[:5],
        "resampling_applied": resampling_info,
        "split_strategy": split_info,
        "model_path": "",
        "features_used": len(feature_cols),
        "test_set_size": len(y_test),
        "test_set_role": "external_holdout" if test_data_path else "internal_test_split",
        "class_labels": _to_jsonable_list(list(pd.Series(y_train).sort_values().unique())),
        "training_backend": "sklearn_fallback",
        "holdout_missing_features_filled": missing_features,
    }, ensure_ascii=False, indent=2)


def train_with_autogluon(
    data_path: str,
    target_column: str,
    balancing_method: str = "none",
    time_limit: int = 180,
    eval_metric: str = "roc_auc",
    presets: str = "best_quality",
    test_data_path: Optional[str] = None
) -> str:
    """
    Train models using AutoGluon with strict train/test splitting and leakage-free resampling.
    
    Logic:
    1. Load Data (Strict Feature Adherence).
    2. Split into Train (80%) and Test (20%) (Stratified).
    3. Apply Resampling (SMOTE+NearMiss) ONLY to Train data if requested.
    4. Train AutoGluon on (Resampled) Train data.
    5. Evaluate on Clean Test data.
    
    Args:
        data_path: Path to the selected dataset (CSV/Excel).
        target_column: Name of the target variable.
        balancing_method: 'none' or 'smote_nearmiss'.
        time_limit: Training time budget in seconds.
        eval_metric: Metric to optimize (e.g., 'roc_auc', 'f1', 'accuracy').
        presets: AutoGluon preset ('best_quality', 'high_quality', 'medium_quality').
        test_data_path: Optional external holdout dataset. If provided, the model
            trains on `data_path` and evaluates on this clean holdout directly.
    
    Returns:
        JSON string with validation-selected champion, holdout metrics, and file paths.
    """
    if not SKLEARN_AVAILABLE and not AUTOGLUON_AVAILABLE:
        return json.dumps({"error": "Required ML libraries are not installed."})
    
    start_time = time.time()
    
    try:
        # 1. Load Data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)
        
        # Feature Stability: We use the columns EXACTLY as they are.
        # No extra feature selection steps here.
        feature_cols = [c for c in df.columns if c != target_column]
        print(f"Model Building: Training on fixed feature set of {len(feature_cols)} features.")
        
        # 2. Strict Train/Test Split
        # If an external holdout is provided, never split the training matrix again.
        if test_data_path:
            if test_data_path.endswith('.csv'):
                holdout_df = pd.read_csv(test_data_path)
            else:
                holdout_df = pd.read_excel(test_data_path)

            if target_column not in holdout_df.columns:
                return json.dumps({
                    "success": False,
                    "error": f"Target column '{target_column}' not found in holdout dataset"
                })

            aligned_holdout_df, missing_features = _align_external_holdout(df, holdout_df, target_column)
            if missing_features:
                print(
                    "Warning: Holdout dataset is missing training features; "
                    f"filling {len(missing_features)} columns with 0.0: {missing_features[:10]}"
                )

            extra_train_cols = [c for c in df.columns if c != target_column]
            X_train = df[extra_train_cols].copy()
            y_train = df[target_column].copy()
            X_test = aligned_holdout_df[extra_train_cols].copy()
            y_test = aligned_holdout_df[target_column].copy()
            split_info = f"external_holdout:{test_data_path}"
        else:
            # random_state ensures reproducibility
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # 80/20 Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            split_info = "internal_train_test_split"
        
        # 3. Conditional Resampling (Train Set ONLY)
        resampling_info = "None"
        if balancing_method == 'smote_nearmiss':
            try:
                if not IMBLEARN_AVAILABLE:
                    raise ImportError("imblearn is not installed")
                # Check for categorical features (SMOTE limitation)
                # We assume Stage 1-5 output is numeric. If not, we skip SMOTE to avoid crash.
                numeric_cols = X_train.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < len(X_train.columns):
                    print("Warning: Categorical features detected. Skipping SMOTE to avoid encoding errors.")
                    train_data = X_train.copy()
                    train_data[target_column] = y_train
                    resampling_info = "Skipped (Categorical Data)"
                else:
                    print("Applying SMOTE + NearMiss to Training Set...")
                    smote = SMOTE(random_state=42)
                    nearmiss = NearMiss(version=3)
                    
                    # Pipeline: Over-sample then Under-sample
                    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                    X_train_res, y_train_res = nearmiss.fit_resample(X_train_res, y_train_res)
                    
                    # Update training data
                    train_data = pd.DataFrame(X_train_res, columns=feature_cols)
                    train_data[target_column] = y_train_res
                    resampling_info = f"SMOTE+NearMiss (Train size: {len(X_train)} -> {len(train_data)})"
            except Exception as e:
                print(f"Resampling failed ({str(e)}), reverting to original training data.")
                train_data = X_train.copy()
                train_data[target_column] = y_train
        else:
            train_data = X_train.copy()
            train_data[target_column] = y_train
        
        # Prepare Test Data (Clean - Never Resampled)
        test_data = X_test.copy()
        test_data[target_column] = y_test
        
        # 4. AutoGluon Training or sklearn fallback
        if not AUTOGLUON_AVAILABLE:
            print("AutoGluon not installed, falling back to sklearn models.")
            return _train_with_sklearn_fallback(
                train_data=train_data,
                test_data=test_data,
                target_column=target_column,
                eval_metric=eval_metric,
                split_info=split_info,
                resampling_info=resampling_info,
                test_data_path=test_data_path,
                missing_holdout_features=missing_features if test_data_path else None,
            )

        model_root_dir = _resolve_autogluon_models_dir()
        os.makedirs(model_root_dir, exist_ok=True)
        # Retries can occur within the same second; avoid reopening a partially fitted learner.
        model_save_path = os.path.join(
            model_root_dir,
            f"ag-{time.time_ns()}-{uuid.uuid4().hex[:8]}",
        )

        # AutoGluon otherwise consumes every visible CPU. On large shared hosts
        # this can create very large joblib thread pools during leaderboard
        # evaluation and fail inside a resource-limited executor. Keep the
        # default conservative while allowing an explicit deployment override.
        detected_cpus = os.cpu_count() or 1
        configured_cpus = int(os.getenv("METABO_AUTOGLUON_NUM_CPUS", "1"))
        autogluon_num_cpus = max(1, min(configured_cpus, detected_cpus))

        # Bagging plus dynamic stacking in the quality presets can consume a
        # short budget before a single complete model exists. Use the standard
        # non-bagged preset for short interactive jobs and report the effective
        # choice in the returned metadata.
        requested_presets = str(presets or "best_quality")
        effective_presets = requested_presets
        if int(time_limit) < 180 and requested_presets in {
            "best_quality",
            "high_quality",
            "good_quality",
        }:
            effective_presets = "medium_quality"
            print(
                "AutoGluon budget guard: "
                f"preset {requested_presets!r} requires a longer budget; "
                f"using {effective_presets!r} for time_limit={time_limit}s."
            )
        print(f"AutoGluon resource guard: num_cpus={autogluon_num_cpus}.")
        
        # Ensure we don't prune features implicitly by setting specific kwargs if needed
        # AutoGluon default behavior is usually good, but we rely on Stage 5 input.
        predictor = TabularPredictor(
            label=target_column,
            eval_metric=eval_metric,
            path=model_save_path,
            verbosity=2
        ).fit(
            train_data=train_data,
            time_limit=time_limit,
            presets=effective_presets,
            num_cpus=autogluon_num_cpus,
            save_space=True,
            excluded_model_types=['KNN']  # KNN is slow on high-dim data
        )

        # 5. Champion Selection and Holdout Evaluation
        # Use train-internal validation to select the champion model so the
        # external holdout remains a final unbiased evaluation set.
        validation_leaderboard = predictor.leaderboard(silent=True)
        holdout_leaderboard = predictor.leaderboard(test_data, silent=True)

        if validation_leaderboard.empty:
            return json.dumps({
                "success": False,
                "error": "AutoGluon returned an empty validation leaderboard"
            })

        best_model_name = validation_leaderboard.iloc[0]['model']
        champion_holdout_row = holdout_leaderboard[holdout_leaderboard['model'] == best_model_name]
        champion_holdout_score = None
        if not champion_holdout_row.empty and 'score_test' in champion_holdout_row.columns:
            champion_holdout_score = _safe_scalar(champion_holdout_row.iloc[0]['score_test'])

        # Detailed metrics for the validation-selected champion on the final test set.
        y_pred = predictor.predict(test_data, model=best_model_name)
        y_prob = predictor.predict_proba(test_data, model=best_model_name)
        
        # Handle multiclass probability shape if necessary (AutoGluon handles this, but metrics need specific format)
        if len(predictor.class_labels) == 2:
            if isinstance(y_prob, pd.DataFrame):
                pos_label = predictor.class_labels[1]
                y_prob_val = y_prob[pos_label]
            else:
                y_prob_val = y_prob[:, 1]
            auc_score = round(roc_auc_score(y_test, y_prob_val), 4)
        else:
            auc_score = "N/A (Multiclass)"
        
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc": auc_score,
            "f1": round(f1_score(y_test, y_pred, average='weighted'), 4),
            "mcc": round(matthews_corrcoef(y_test, y_pred), 4)
        }
        
        # ========================================================================
        # Phase 3 Enhancement: Extract Top 5 Models' Predictions for ROC Plotting
        # ========================================================================
        print("\n[Phase 3 Data Export] Extracting Top 5 models' predictions for visualization...")
        
        leaderboard_with_predictions = []
        top5_models = validation_leaderboard.head(5)
        
        for idx, row in top5_models.iterrows():
            model_name = row['model']
            score_val = _safe_scalar(row.get('score_val'))
            holdout_row = holdout_leaderboard[holdout_leaderboard['model'] == model_name]
            score_holdout = None
            if not holdout_row.empty and 'score_test' in holdout_row.columns:
                score_holdout = _safe_scalar(holdout_row.iloc[0]['score_test'])
            
            try:
                # Get predictions from this specific model
                y_prob_model = predictor.predict_proba(test_data, model=model_name)
                
                # Extract positive class probability for binary classification
                if len(predictor.class_labels) == 2:
                    if isinstance(y_prob_model, pd.DataFrame):
                        pos_label = predictor.class_labels[1]
                        predictions = y_prob_model[pos_label].tolist()
                    else:
                        predictions = y_prob_model[:, 1].tolist()
                else:
                    # For multiclass, store all probabilities
                    if isinstance(y_prob_model, pd.DataFrame):
                        predictions = y_prob_model.values.tolist()
                    else:
                        predictions = y_prob_model.tolist()
                
                leaderboard_with_predictions.append({
                    'model': model_name,
                    'score_val': score_val,
                    'score_holdout': score_holdout,
                    'predictions': predictions  # ← 关键：真实预测概率
                })
                
                holdout_msg = f"{score_holdout:.4f}" if isinstance(score_holdout, float) else "N/A"
                val_msg = f"{score_val:.4f}" if isinstance(score_val, float) else "N/A"
                print(
                    f"  ✓ {model_name}: val={val_msg}, holdout={holdout_msg}, "
                    f"predictions_shape={len(predictions)}"
                )
                
            except Exception as e:
                print(f"  ⚠ Warning: Failed to extract predictions for {model_name}: {str(e)}")
                # Fallback: 保存不带预测的记录
                leaderboard_with_predictions.append({
                    'model': model_name,
                    'score_val': score_val,
                    'score_holdout': score_holdout,
                    'predictions': None
                })
        
        print(f"[Phase 3 Data Export] Successfully extracted predictions for {len(leaderboard_with_predictions)} models")
        
        # ========================================================================
        # Return Enhanced Results
        # ========================================================================
        
        # CRITICAL FIX: Handle class_labels conversion safely
        # predictor.class_labels might already be a list or a numpy array
        if isinstance(predictor.class_labels, list):
            class_labels_list = predictor.class_labels
        elif hasattr(predictor.class_labels, 'tolist'):
            class_labels_list = predictor.class_labels.tolist()
        else:
            class_labels_list = list(predictor.class_labels)
        
        return json.dumps({
            "success": True,
            "best_model": best_model_name,
            "best_model_selection_basis": "train_internal_validation",
            "best_model_validation_score": _safe_scalar(validation_leaderboard.iloc[0].get('score_val')),
            "best_model_holdout_score": champion_holdout_score,
            "primary_metric": eval_metric,
            "test_set_metrics": metrics,
            "leaderboard_top5": leaderboard_with_predictions,  # 按内部验证排序，附带 holdout 表现
            "resampling_applied": resampling_info,
            "split_strategy": split_info,
            "model_path": model_save_path,
            "requested_presets": requested_presets,
            "effective_presets": effective_presets,
            "num_cpus": autogluon_num_cpus,
            "features_used": len(feature_cols),
            "test_set_size": len(y_test),  # 新增：测试集大小
            "test_set_role": "external_holdout" if test_data_path else "internal_test_split",
            "class_labels": class_labels_list  # 修复：安全转换为 list
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        import traceback
        error_text = str(e)
        if "learner is already fit" in error_text.lower() and SKLEARN_AVAILABLE:
            try:
                return _train_with_sklearn_fallback(
                    train_data=train_data,
                    test_data=test_data,
                    target_column=target_column,
                    eval_metric=eval_metric,
                    split_info=split_info + "|autogluon_recovery",
                    resampling_info=resampling_info,
                    test_data_path=test_data_path,
                    missing_holdout_features=missing_features if test_data_path else None,
                )
            except Exception as fallback_error:
                error_text = f"{error_text}; sklearn recovery failed: {fallback_error}"
        return json.dumps({"error": f"Training failed: {error_text}", "traceback": traceback.format_exc()})


def _resolve_autogluon_models_dir() -> str:
    """
    Resolve AutoGluon model root directory from config with safe fallbacks.
    Never fall back to repository-root relative `./AutogluonModels`.
    """
    try:
        from src.utils.config_manager import get_config
        cfg = get_config()
        phase1_paths = cfg.get_phase1_paths()

        configured = phase1_paths.get("autogluon_models_dir")
        if configured:
            return configured

        artifacts_dir = phase1_paths.get("artifacts_dir")
        if artifacts_dir:
            return os.path.join(artifacts_dir, "models", "autogluon")
    except Exception:
        pass

    return "output/phase1/artifacts/models/autogluon"


def _resolve_model_results_output_dir() -> str:
    """
    Resolve model-results output directory from config with safe fallbacks.
    Never fall back to a repository-root relative `./model_results`.
    """
    try:
        from src.utils.config_manager import get_config
        cfg = get_config()
        phase1_paths = cfg.get_phase1_paths()

        # Preferred: explicit key if added in config.yaml
        configured = phase1_paths.get("model_results_dir")
        if configured:
            return configured

        # Fallback 1: place under phase1 artifacts root
        artifacts_dir = phase1_paths.get("artifacts_dir")
        if artifacts_dir:
            return os.path.join(artifacts_dir, "model_results")
    except Exception:
        # Keep a non-root fallback path if config is unavailable
        pass

    # Fallback 2: canonical non-root location
    return "output/phase1/artifacts/model_results"


def _resolve_autogluon_results_paths() -> List[str]:
    """Resolve canonical and legacy AutoGluon results paths with de-duplication."""
    candidates: List[str] = []
    try:
        from src.utils.config_manager import get_config
        cfg = get_config()
        phase1_paths = cfg.get_phase1_paths()
        candidates.extend([
            os.path.join(
                phase1_paths.get("artifacts_dir") or "output/phase1/artifacts",
                "autogluon_training_results.json",
            ),
            phase1_paths.get("autogluon_results") or "data/autogluon_training_results.json",
        ])
    except Exception:
        candidates.extend([
            "output/phase1/artifacts/autogluon_training_results.json",
            "data/autogluon_training_results.json",
        ])

    resolved: List[str] = []
    seen = set()
    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        if abs_candidate in seen:
            continue
        resolved.append(abs_candidate)
        seen.add(abs_candidate)
    return resolved


def save_model_results(
    train_result_json: str,
    output_dir: Optional[str] = None
) -> str:
    """
    Save the training report and leaderboard to disk.
    """
    try:
        resolved_output_dir = output_dir or _resolve_model_results_output_dir()
        os.makedirs(resolved_output_dir, exist_ok=True)
        data = json.loads(train_result_json)
        
        if "error" in data:
            return json.dumps({"error": "Cannot save failed run"})

        # Persist the canonical AutoGluon results artifact first so downstream
        # consumers never keep reading a stale JSON from previous runs.
        ag_results_paths = _resolve_autogluon_results_paths()
        for ag_results_path in ag_results_paths:
            os.makedirs(os.path.dirname(ag_results_path), exist_ok=True)
            with open(ag_results_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save JSON Report
        report_path = os.path.join(resolved_output_dir, "modeling_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save Leaderboard if available
        lb_path = None
        if "leaderboard_top5" in data:
            lb_df = pd.DataFrame(data["leaderboard_top5"])
            lb_path = os.path.join(resolved_output_dir, "leaderboard_top5.csv")
            lb_df.to_csv(lb_path, index=False)
        
        return json.dumps({
            "success": True,
            "autogluon_results_paths": ag_results_paths,
            "report_path": report_path,
            "leaderboard_path": lb_path,
            "message": "Results saved successfully."
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# LangChain Tool Wrappers
# ============================================================================
from langchain_core.tools import StructuredTool

# Create tool wrappers for LangChain/LangGraph
analyze_data_for_modeling_tool = StructuredTool.from_function(
    func=analyze_data_for_modeling,
    name="analyze_data_for_modeling",
    description=analyze_data_for_modeling.__doc__
)

train_with_autogluon_tool = StructuredTool.from_function(
    func=train_with_autogluon,
    name="train_with_autogluon",
    description=train_with_autogluon.__doc__
)

save_model_results_tool = StructuredTool.from_function(
    func=save_model_results,
    name="save_model_results",
    description=save_model_results.__doc__
)

# Export tools (using wrappers)
MODEL_BUILDING_TOOLS = [
    analyze_data_for_modeling_tool,
    train_with_autogluon_tool,
    save_model_results_tool
]
