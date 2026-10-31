from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import json

@dataclass
class PipelineConfig:
    # --- Experiment Metadata ---
    project_name: str = "tabular_experiment"
    output_dir: str = "experiments/run_latest"
    seed: int = 42

    # --- Data Definition ---
    id_col: Optional[str] = None
    target_col: str = "target"
    group_col: Optional[str] = None      # Used if cv_strategy is group_kfold or stratified_group_kfold
    time_col: Optional[str] = None       # Used if cv_strategy is time_series
    drop_cols: List[str] = field(default_factory=list)

    # --- Task & Evaluation ---
    # task_type: "binary" | "multiclass" | "regression"
    task_type: str = "binary"
    # metric: "roc_auc" | "accuracy" | "f1_macro" | "f1_binary" | "logloss" | "rmse" | "mae" | "r2" | "qwk"
    eval_metric: str = "roc_auc"
    optimize_threshold: bool = True     # For binary classification (tunes decision threshold on OOF)

    # --- Validation Strategy ---
    # cv_strategy: "stratified_kfold" | "kfold" | "group_kfold" | "stratified_group_kfold" | "time_series"
    cv_strategy: str = "stratified_kfold"
    n_splits: int = 5
    shuffle: bool = True

    # --- Feature Engineering ---
    run_generic_features: bool = True     # Auto-create frequency, aggregations, missing flags
    auto_fe_preset: str = "conservative" # "conservative" | "aggressive" | "off"
    auto_fe_max_features: int = 30
    create_missing_flags: bool = True
    create_freq_encoding: bool = True
    create_group_aggregations: bool = True
    max_agg_cats: int = 3                # Limit number of categorical cols used for group aggregations
    max_agg_nums: int = 5                # Limit number of numerical cols aggregated

    # --- Preprocessing & Encoders ---
    # categorical_encoder: "ordinal" | "target" | "none" (tree models can handle ordinal/native)
    categorical_encoder: str = "ordinal"
    handle_missing: str = "auto"          # "auto" (keeps NaN for GBDT) or "impute" (for linear/neural)

    # --- Diagnostics ---
    run_adversarial_validation: bool = True
    adversarial_auc_threshold: float = 0.70  # Warn if train-test AUC > threshold

    # --- Models to Train ---
    # Options: "lgbm", "xgb", "catboost", "rf", "ridge", "logreg"
    models: List[str] = field(default_factory=lambda: ["lgbm", "xgb", "catboost"])
    model_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- Ensembling ---
    ensemble_methods: List[str] = field(default_factory=lambda: ["mean", "rank", "optimal_weights", "stacking", "hill_climbing"])
    save_oof_predictions: bool = True
    save_test_predictions: bool = True

    def validate(self):
        valid_tasks = ["binary", "multiclass", "regression"]
        if self.task_type not in valid_tasks:
            raise ValueError(f"task_type must be one of {valid_tasks}, got {self.task_type}")
        
        valid_cv = ["stratified_kfold", "kfold", "group_kfold", "stratified_group_kfold", "time_series"]
        if self.cv_strategy not in valid_cv:
            raise ValueError(f"cv_strategy must be one of {valid_cv}, got {self.cv_strategy}")

        if "group" in self.cv_strategy and not self.group_col:
            raise ValueError(f"cv_strategy '{self.cv_strategy}' requires group_col to be specified.")

        if self.cv_strategy == "time_series" and not self.time_col:
            raise ValueError("cv_strategy 'time_series' requires time_col to be specified.")

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {k: v for k, v in self.__dict__.items()}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "PipelineConfig":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)
