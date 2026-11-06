import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tabular_pipeline import PipelineConfig, ModelFactory, IndoBERTEngine


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
        # Either device='cuda' (xgb >= 2.0) or tree_method='gpu_hist'
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


if __name__ == "__main__":
    unittest.main()
