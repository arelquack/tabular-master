import os
import json
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, Callable

from .config import PipelineConfig
from .data_detector import DataDetector
from .feature_engine import BaseFeatureHook, UniversalAutoFeatureEngine, GenericFeatureEngine
from .diagnostics import AdversarialValidator
from .trainer import CrossValidatorTrainer
from .ensemble import EnsembleOptimizer
from .metrics import MetricEvaluator

class TabularPipeline:
    """
    End-to-End Modular Tabular Data Mining Framework.
    Automates data diagnosis, feature engineering, cross-validation,
    multi-model training, ensembling, threshold tuning, and submission generation.
    """
    def __init__(
        self,
        config: PipelineConfig,
        feature_hook: Optional[Union[BaseFeatureHook, Callable]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        is_aborted: Optional[Callable[[], bool]] = None
    ):
        self.config = config
        self.config.validate()
        self.feature_hook = feature_hook
        self.progress_callback = progress_callback
        self.is_aborted = is_aborted
        self.detector = DataDetector()
        self.results: Dict[str, Any] = {}
        os.makedirs(self.config.output_dir, exist_ok=True)

    def fit_predict(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Alias for run() to support standard scikit-learn/AutoML style API."""
        return self.run(train_df, test_df)

    def run(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        start_time = time.time()

        # Seed everything for 100% reproducibility
        import random
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        os.environ["PYTHONHASHSEED"] = str(self.config.seed)

        print("=" * 60)
        print(f"[START] TABULAR PIPELINE: {self.config.project_name.upper()}")
        print(f"Task: {self.config.task_type} | Metric: {self.config.eval_metric} | CV: {self.config.cv_strategy} ({self.config.n_splits} folds)")
        print("=" * 60)

        train = train_df.copy()
        test = test_df.copy() if test_df is not None else None

        # 1. Target and ID Separation
        target_col = self.config.target_col
        id_col = self.config.id_col
        group_col = self.config.group_col
        time_col = self.config.time_col

        if target_col not in train.columns:
            raise ValueError(f"Target column '{target_col}' not found in train dataframe.")

        y_raw = train[target_col].values
        test_ids = test[id_col].values if (test is not None and id_col and id_col in test.columns) else None
        groups = train[group_col] if (group_col and group_col in train.columns) else None

        # Convert target to numerical format if boolean or string
        if self.config.task_type == "binary":
            if y_raw.dtype == bool:
                y_train = y_raw.astype(int)
            elif not pd.api.types.is_numeric_dtype(pd.Series(y_raw)):
                y_train = pd.Series(y_raw).astype("category").cat.codes.values
            else:
                y_train = y_raw.astype(int)
        elif self.config.task_type == "multiclass":
            if not pd.api.types.is_numeric_dtype(pd.Series(y_raw)):
                y_train = pd.Series(y_raw).astype("category").cat.codes.values
            else:
                y_train = y_raw.astype(int)
        else:
            y_train = y_raw.astype(float)

        # Separate features from target
        train = train.drop(columns=[target_col])

        # 2. Domain Knowledge Feature Hook (User-defined)
        if self.feature_hook is not None:
            print("\n[Step 1/6] Running Domain Feature Engineering Hook...")
            if isinstance(self.feature_hook, BaseFeatureHook):
                train, test = self.feature_hook.transform(train, test)
            elif callable(self.feature_hook):
                train, test = self.feature_hook(train, test)

        # 3. Data Type Inspection
        print("\n[Step 2/6] Inspecting Data Types & Schemas...")
        drop_cols = [c for c in [id_col, group_col, time_col] + self.config.drop_cols if c is not None]
        detected = self.detector.inspect(train, exclude_cols=drop_cols)
        print(f"  Numeric: {len(detected['numeric'])} cols")
        print(f"  Categorical: {len(detected['all_cat'])} cols (Low: {len(detected['low_cardinality_cat'])}, High: {len(detected['high_cardinality_cat'])})")
        print(f"  Datetime: {len(detected['datetime'])} cols")

        # 4. Adversarial Validation Check
        feature_cols_initial = [c for c in train.columns if c not in drop_cols]
        if test is not None and self.config.run_adversarial_validation:
            print("\n[Step 3/6] Running Adversarial Validation (Train vs Test Drift)...")
            adv_validator = AdversarialValidator(n_splits=3, seed=self.config.seed)
            adv_results = adv_validator.check(train, test, feature_cols=feature_cols_initial)
            print(f"  Adversarial AUC: {adv_results['adversarial_auc']:.4f} -> {adv_results['status']}")
            if adv_results["is_shifted"]:
                print("  Top 3 Drifting Features:")
                for item in adv_results["top_drift_features"][:3]:
                    print(f"    - {item['feature']}: importance {item['importance']:.4f}")
            self.results["adversarial_validation"] = adv_results

        # 5. Universal Automated Feature Engineering (Auto-FE)
        if self.config.run_generic_features and self.config.auto_fe_preset != "off":
            print(f"\n[Step 4/6] Running Universal Auto-FE (Preset: {self.config.auto_fe_preset.upper()})...")
            if self.progress_callback:
                self.progress_callback({
                    "stage": "auto_fe",
                    "message": f"Running Universal Auto-FE (Preset: {self.config.auto_fe_preset.upper()})..."
                })
            auto_engine = UniversalAutoFeatureEngine(
                preset=self.config.auto_fe_preset,
                max_features=self.config.auto_fe_max_features,
            )
            train, test = auto_engine.generate(train, test, detected)

        # Prepare final feature matrices
        # Sanitize column names for XGBoost / LightGBM compatibility
        import re
        clean_name_map = {c: re.sub(r'[\[\]<>, "]+', '_', str(c)) for c in train.columns}
        train = train.rename(columns=clean_name_map)
        if test is not None:
            test = test.rename(columns=clean_name_map)

        drop_cols_clean = [clean_name_map.get(c, c) for c in drop_cols]
        feature_cols = [c for c in train.columns if c not in drop_cols_clean]
        X_train = train[feature_cols].copy()
        X_test = test[feature_cols].copy() if test is not None else None
        print(f"  Total Final Features for Training: {X_train.shape[1]}")

        # Re-detect categorical columns for encoder
        updated_detected = self.detector.inspect(X_train)
        cat_cols = updated_detected["all_cat"]

        # 6. Multi-Model Cross-Validation Training
        print("\n[Step 5/6] Training Models with Leak-Free Cross-Validation...")
        trainer = CrossValidatorTrainer(
            task_type=self.config.task_type,
            eval_metric=self.config.eval_metric,
            cv_strategy=self.config.cv_strategy,
            n_splits=self.config.n_splits,
            categorical_encoder=self.config.categorical_encoder,
            seed=self.config.seed,
            use_gpu=self.config.use_gpu,
            progress_callback=self.progress_callback,
            is_aborted=self.is_aborted,
        )

        model_results = {}
        for m_name in self.config.models:
            if self.is_aborted and self.is_aborted():
                raise InterruptedError("Execution stopped by user.")
            m_params = self.config.model_params.get(m_name, {})
            res = trainer.train_model(
                model_name=m_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                groups=groups,
                cat_cols=cat_cols,
                model_params=m_params,
                early_stopping_rounds=100,
                verbose=False,
            )
            model_results[m_name] = res

            # Cache OOF and Test predictions as .npy artifacts for fast reload/ensemble
            artifacts_dir = os.path.join(self.config.output_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            np.save(os.path.join(artifacts_dir, f"oof_{m_name}.npy"), res["oof_preds"])
            if res["test_preds"] is not None:
                np.save(os.path.join(artifacts_dir, f"test_{m_name}.npy"), res["test_preds"])

        self.results["models"] = model_results

        # 7. Ensembling & Blending
        print("\n[Step 6/6] Ensembling & Post-Processing...")
        ensemble_opt = EnsembleOptimizer(task_type=self.config.task_type, eval_metric=self.config.eval_metric)
        model_names = list(model_results.keys())
        oofs_list = [model_results[m]["oof_preds"] for m in model_names]
        tests_list = [model_results[m]["test_preds"] for m in model_names] if test is not None else []

        ensemble_scores = {}
        best_ensemble_name = None
        best_ensemble_score = -float("inf") if MetricEvaluator.is_higher_better(self.config.eval_metric) else float("inf")
        best_oof_preds = None
        best_test_preds = None

        def is_better(score, best_score):
            if MetricEvaluator.is_higher_better(self.config.eval_metric):
                return score > best_score
            return score < best_score

        # Check individual models first
        for m in model_names:
            score = model_results[m]["overall_score"]
            ensemble_scores[m] = score
            if is_better(score, best_ensemble_score):
                best_ensemble_score = score
                best_ensemble_name = m
                best_oof_preds = model_results[m]["oof_preds"]
                best_test_preds = model_results[m]["test_preds"] if test is not None else None

        if len(model_names) > 1:
            if self.progress_callback is not None:
                self.progress_callback({
                    "stage": "ensembling",
                    "message": "Evaluating ensembling strategies (Simple Mean, Rank, SLSQP, Stacking, Caruana)..."
                })
            # 1. Simple Mean Blend
            mean_oof = ensemble_opt.simple_average(oofs_list)
            mean_score = MetricEvaluator.calculate(y_train, mean_oof, self.config.eval_metric, self.config.task_type)
            ensemble_scores["ensemble_mean"] = mean_score
            print(f"  - Simple Average Blend Score: {mean_score:.5f}")
            if is_better(mean_score, best_ensemble_score):
                best_ensemble_score = mean_score
                best_ensemble_name = "ensemble_mean"
                best_oof_preds = mean_oof
                best_test_preds = ensemble_opt.simple_average(tests_list) if test is not None else None

            # 1b. Rank Average Blend (Particularly powerful for AUC / ranking metrics)
            if "auc" in self.config.eval_metric.lower():
                rank_oof = ensemble_opt.rank_average(oofs_list)
                rank_score = MetricEvaluator.calculate(y_train, rank_oof, self.config.eval_metric, self.config.task_type)
                ensemble_scores["ensemble_rank"] = rank_score
                print(f"  - Rank Average Blend Score: {rank_score:.5f}")
                if is_better(rank_score, best_ensemble_score):
                    best_ensemble_score = rank_score
                    best_ensemble_name = "ensemble_rank"
                    best_oof_preds = rank_oof
                    best_test_preds = ensemble_opt.rank_average(tests_list) if test is not None else None

            # 2. Optimal Weighted Blend
            opt_weights = ensemble_opt.find_optimal_weights(y_train, oofs_list)
            opt_oof = sum(w * oof for w, oof in zip(opt_weights, oofs_list))
            opt_score = MetricEvaluator.calculate(y_train, opt_oof, self.config.eval_metric, self.config.task_type)
            ensemble_scores["ensemble_opt_weights"] = opt_score
            print(f"  - Optimal Weighted Blend Score: {opt_score:.5f} (Weights: {dict(zip(model_names, np.round(opt_weights, 3)))})")
            if is_better(opt_score, best_ensemble_score):
                best_ensemble_score = opt_score
                best_ensemble_name = "ensemble_opt_weights"
                best_oof_preds = opt_oof
                best_test_preds = sum(w * t for w, t in zip(opt_weights, tests_list)) if test is not None else None

            # 3. Stacking
            oof_dict = {m: model_results[m]["oof_preds"] for m in model_names}
            test_dict = {m: model_results[m]["test_preds"] for m in model_names} if test is not None else None
            stack_oof, stack_test = ensemble_opt.stack(y_train, oof_dict, test_dict)
            stack_score = MetricEvaluator.calculate(y_train, stack_oof, self.config.eval_metric, self.config.task_type)
            ensemble_scores["ensemble_stacking"] = stack_score
            print(f"  - Stacking Meta-Learner Score: {stack_score:.5f}")
            if is_better(stack_score, best_ensemble_score):
                best_ensemble_score = stack_score
                best_ensemble_name = "ensemble_stacking"
                best_oof_preds = stack_oof
                best_test_preds = stack_test

            # 4. Caruana's Hill Climbing (Greedy Forward Ensemble with Replacement)
            hill_weights, hill_oof, hill_score = ensemble_opt.hill_climbing(y_train, oof_dict)
            ensemble_scores["ensemble_hill_climbing"] = hill_score
            active_weights = {k: round(v, 3) for k, v in hill_weights.items()}
            print(f"  - Caruana Hill Climbing Score: {hill_score:.5f} (Weights: {active_weights})")
            if is_better(hill_score, best_ensemble_score):
                best_ensemble_score = hill_score
                best_ensemble_name = "ensemble_hill_climbing"
                best_oof_preds = hill_oof
                if test is not None and test_dict is not None:
                    best_test_preds = sum(w * test_dict[k] for k, w in hill_weights.items())

        print(f"\n[CHAMPION] CHAMPION STRATEGY: '{best_ensemble_name}' with {self.config.eval_metric}: {best_ensemble_score:.5f}")

        # 8. Threshold Optimization (Binary Classification)
        best_threshold = 0.50
        is_auc_metric = "auc" in self.config.eval_metric.lower()
        if self.config.task_type == "binary" and self.config.optimize_threshold and not is_auc_metric:
            th, th_score = MetricEvaluator.optimize_binary_threshold(
                y_train, best_oof_preds, self.config.eval_metric
            )
            print(f"  Optimal Decision Threshold: {th:.4f} (Score: {th_score:.5f})")
            best_threshold = th
            self.results["best_threshold"] = best_threshold

        # 9. Generate Submission & Save Outputs
        if test is not None and best_test_preds is not None:
            sub_df = pd.DataFrame()
            if id_col and test_ids is not None:
                sub_df[id_col] = test_ids
            else:
                sub_df["id"] = np.arange(len(test))

            if self.config.task_type == "binary":
                if is_auc_metric:
                    # Competitions evaluated on ROC-AUC require continuous probability predictions
                    sub_df[target_col] = best_test_preds
                else:
                    is_target_bool = train_df[target_col].dtype == bool
                    binary_preds = (best_test_preds >= best_threshold)
                    sub_df[target_col] = binary_preds if is_target_bool else binary_preds.astype(int)
            elif self.config.task_type == "multiclass":
                sub_df[target_col] = np.argmax(best_test_preds, axis=1)
            else:
                sub_df[target_col] = best_test_preds

            sub_path = os.path.join(self.config.output_dir, "submission.csv")
            sub_df.to_csv(sub_path, index=False)
            print(f"\n[OUTPUT] Saved Submission: {sub_path} (Shape: {sub_df.shape})")

        # Save OOF and metadata
        oof_df = pd.DataFrame({"target": y_train, "oof_pred": best_oof_preds})
        oof_path = os.path.join(self.config.output_dir, "oof_champion.csv")
        oof_df.to_csv(oof_path, index=False)
        np.save(os.path.join(self.config.output_dir, "artifacts", "y_true.npy"), y_train)

        # Save Top Feature Importance across models
        top_importances = []
        for m in model_names:
            fi = model_results[m]["importance_df"].copy()
            fi["model"] = m
            top_importances.append(fi)
        if top_importances:
            all_fi = pd.concat(top_importances, ignore_index=True)
            fi_path = os.path.join(self.config.output_dir, "feature_importances.csv")
            all_fi.to_csv(fi_path, index=False)

        # Save Config & Summary
        self.config.save(os.path.join(self.config.output_dir, "config.json"))
        summary = {
            "project_name": self.config.project_name,
            "champion_model": best_ensemble_name,
            "champion_score": best_ensemble_score,
            "metric": self.config.eval_metric,
            "best_threshold": best_threshold,
            "all_scores": ensemble_scores,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        with open(os.path.join(self.config.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        print(f"[DONE] Finished in {summary['elapsed_seconds']}s. All artifacts saved to '{self.config.output_dir}'")
        self.results.update(summary)
        return self.results

    @classmethod
    def load_cached_predictions(cls, output_dir: str, model_names: list[str]):
        """
        Loads pre-computed OOF and Test predictions from .npy files.
        Enables instantaneous re-ensembling without retraining!
        """
        artifacts_dir = os.path.join(output_dir, "artifacts")
        oofs = {m: np.load(os.path.join(artifacts_dir, f"oof_{m}.npy")) for m in model_names}
        test_files = [os.path.join(artifacts_dir, f"test_{m}.npy") for m in model_names]
        tests = {m: np.load(f) for m, f in zip(model_names, test_files) if os.path.exists(f)}
        return oofs, tests
