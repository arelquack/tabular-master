import unittest
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tabular_pipeline import PipelineConfig, ModelFactory, IndoBERTEngine
from tabular_pipeline.text_engine import UniversalTextCleaner, TextFeatureExtractor, NLPBaselineTrainer
from train_transformer_universal import resolve_model_path


class TabularGpuAndPipelineTest(unittest.TestCase):
    def test_pipeline_config_gpu_flag(self):
        cfg_cpu = PipelineConfig(use_gpu=False)
        self.assertFalse(cfg_cpu.use_gpu)

        cfg_gpu = PipelineConfig(use_gpu=True)
        self.assertTrue(cfg_gpu.use_gpu)

    def test_model_factory_lgbm_gpu_params(self):
        m_cpu = ModelFactory.create("lgbm", use_gpu=False)
        self.assertNotIn("device", m_cpu._get_default_params())

        m_gpu = ModelFactory.create("lgbm", use_gpu=True)
        params = m_gpu._get_default_params()
        self.assertEqual(params.get("device"), "gpu")
        self.assertNotIn("n_jobs", params)

    def test_model_factory_xgb_gpu_params(self):
        m_cpu = ModelFactory.create("xgb", use_gpu=False)
        self.assertEqual(m_cpu._get_default_params().get("tree_method"), "hist")

        m_gpu = ModelFactory.create("xgb", use_gpu=True)
        params = m_gpu._get_default_params()
        self.assertTrue(params.get("device") == "cuda" or params.get("tree_method") == "gpu_hist")

    def test_model_factory_catboost_gpu_params(self):
        m_cpu = ModelFactory.create("catboost", use_gpu=False)
        self.assertNotIn("task_type", m_cpu._get_default_params())

        m_gpu = ModelFactory.create("catboost", use_gpu=True)
        params = m_gpu._get_default_params()
        self.assertEqual(params.get("task_type"), "GPU")

    def test_indobert_gpu_status(self):
        status = IndoBERTEngine.get_gpu_status()
        self.assertIn("cuda_available", status)
        self.assertIn("device_name", status)
        self.assertIn("details", status)

    def test_text_cleaner_and_stylometrics(self):
        cleaner = UniversalTextCleaner()
        raw = "Halo SEMUANYA!! Cek di https://gemastik.id #semangat @panitia :)"
        cleaned = cleaner.clean(raw)
        self.assertNotIn("https://", cleaned)

        df = pd.DataFrame({"text": [cleaned, "Teks kedua sangat informatif."]})
        features = TextFeatureExtractor.extract_stylometrics(df, "text")
        self.assertIn("txt_word_count", features.columns)
        self.assertIn("txt_char_count", features.columns)

    def test_nlp_baseline_trainer(self):
        df = pd.DataFrame({
            "id": range(1, 21),
            "text": ["layanan sangat bagus sekali dan memuaskan"] * 10 + ["kecewa sekali sistem eror dan lambat"] * 10,
            "sentiment": ["Positif"] * 10 + ["Negatif"] * 10
        })
        res = NLPBaselineTrainer.train_baseline(
            train_df=df,
            text_col="text",
            label_col="sentiment",
            n_splits=2,
            max_features=100
        )
        self.assertIn("overall_accuracy", res)
        self.assertGreater(res["overall_accuracy"], 0.7)

    def test_resolve_model_path(self):
        resolved = resolve_model_path("indobenchmark/indobert-base-p1")
        self.assertTrue(isinstance(resolved, str))

    def test_universal_text_normalizer_and_eda(self):
        import tempfile
        from tabular_pipeline.text_engine import UniversalTextNormalizer, TextEDAReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two partitioned CSV files
            p1 = os.path.join(tmpdir, "part_1.csv")
            p2 = os.path.join(tmpdir, "part_2.csv")
            pd.DataFrame({"doc": ["teks berita positif pertama hari ini", "berita ekonomi bertumbuh pesat"], "target": ["A", "B"]}).to_csv(p1, index=False)
            pd.DataFrame({"doc": ["kabar hoax dan manipulasi informasi", "klarifikasi data dan fakta valid"], "target": ["B", "A"]}).to_csv(p2, index=False)

            out_csv = os.path.join(tmpdir, "normalized_train.csv")
            res = UniversalTextNormalizer.normalize_to_csv(
                source_path=os.path.join(tmpdir, "part_*.csv"),
                output_csv_path=out_csv
            )
            self.assertEqual(res["total_rows"], 4)
            self.assertTrue(os.path.exists(out_csv))

            norm_df = pd.read_csv(out_csv)
            eda_rep = TextEDAReporter.generate_report(norm_df, text_col=res["detected_text_col"], label_col=res["detected_label_col"])
            self.assertIn("length_histogram", eda_rep)
            self.assertIn("class_length_profile", eda_rep)
            self.assertIn("actionable_insights", eda_rep)
            self.assertGreater(len(eda_rep["length_histogram"]), 0)


if __name__ == "__main__":
    unittest.main()
