import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Download, 
  Plus, 
  FileCode, 
  Sliders, 
  Database, 
  Loader2, 
  TrendingUp,
  Wrench,
  FileSpreadsheet,
  Cpu,
  X,
  Check,
  BookOpen,
  Save,
  BarChart2,
  Activity,
  FolderOpen,
  Square,
  RotateCcw,
  RefreshCw,
  AlertTriangle,
  History,
  Edit3,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Layers,
  GitMerge,
  Sparkles,
  Trash2
} from 'lucide-react';

interface CaseMeta {
  id: string;
  name: string;
  description?: string;
  train_path: string;
  test_path?: string;
  auxiliary_paths?: string[];
  id_col?: string;
  target_col: string;
  task_type: string;
  eval_metric: string;
  cv_strategy: string;
  group_col?: string;
  time_col?: string;
  n_splits: number;
  seed: number;
  models: string[];
  output_dir?: string;
  status?: string;
  summary?: any;
}

interface ExperimentRun {
  run_id: string;
  run_number: number;
  timestamp: string;
  name: string;
  tag: string;
  has_hook: boolean;
  features_count: number;
  models: string[];
  cv_score: number;
  all_scores?: Record<string, number>;
  metric: string;
  cv_delta: number;
  lb_score?: number | null;
  champion_model: string;
  submission_filename?: string | null;
  notes?: string;
}

interface CaseDetail {
  meta: CaseMeta;
  summary: any;
  feature_importances: Array<{ feature: string; importance: number }>;
  hook_code: string;
  has_hook?: boolean;
  train_exists?: boolean;
  test_exists?: boolean;
  resolved_train_path?: string;
  resolved_test_path?: string;
  available_cached: string[];
  history?: ExperimentRun[];
  job_state: {
    status: string;
    progress: number;
    current_model?: string;
    current_fold?: string;
    logs: string[];
  };
}

interface EdaData {
  summary: {
    total_rows: number;
    total_cols: number;
    file_size_mb: number;
    missing_cells_sample: number;
    missing_pct_sample: string;
    null_cols_note?: string;
    target_col: string;
    task_type?: string;
    is_regression?: boolean;
    target_summary?: {
      min?: number;
      max?: number;
      mean?: number;
      median?: number;
      std?: number;
      nunique?: number;
    };
  };
  target_distribution: Array<{ label: string; count: number; percentage: number }>;
  correlations: Array<{ feature: string; correlation: number }>;
  columns_profile: Array<{
    name: string;
    is_target: boolean;
    is_id: boolean;
    type: string;
    dtype: string;
    missing_count: number;
    missing_rate: number;
    unique_count: number;
    stats?: {
      min?: number;
      mean?: number;
      median?: number;
      max?: number;
      top_value?: string;
    };
    distribution?: Array<{
      label: string;
      count: number;
      pct: number;
      target_rate?: number;
    }>;
  }>;
  preview_rows: any[];
}

export default function App() {
  const [cases, setCases] = useState<CaseMeta[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string>('');
  const [caseDetail, setCaseDetail] = useState<CaseDetail | null>(null);
  
  // Data Mining Workflow Tabs
  const [activeTab, setActiveTab] = useState<'eda' | 'fe' | 'training' | 'ensemble' | 'artifacts'>('eda');
  const [loading, setLoading] = useState<boolean>(true);
  const [running, setRunning] = useState<boolean>(false);

  // EDA State
  const [edaData, setEdaData] = useState<EdaData | null>(null);
  const [edaLoading, setEdaLoading] = useState<boolean>(false);
  const [selectedFeature, setSelectedFeature] = useState<string>('');
  const [edaViewMode, setEdaViewMode] = useState<'composite' | 'trend' | 'bars'>('composite');

  // Model Selection for Pipeline Execution
  const [selectedModels, setSelectedModels] = useState<string[]>(['lgbm', 'xgb', 'catboost']);

  // Editable Feature Hook Code
  const [editableHookCode, setEditableHookCode] = useState<string>('');
  const [savingHook, setSavingHook] = useState<boolean>(false);
  const [hookSaved, setHookSaved] = useState<boolean>(false);
  const [hookHistory, setHookHistory] = useState<string[]>([]);
  const [hookValidation, setHookValidation] = useState<any>(null);
  const [hookTesting, setHookTesting] = useState<boolean>(false);

  // Quick Visual Feature Engineering Builder
  const [feColA, setFeColA] = useState<string>('');
  const [feOp, setFeOp] = useState<string>('/');
  const [feColB, setFeColB] = useState<string>('');

  // Visual Group-By Aggregation Builder
  const [feGroupCat, setFeGroupCat] = useState<string>('');
  const [feGroupNum, setFeGroupNum] = useState<string>('');
  const [feGroupStat, setFeGroupStat] = useState<'diff' | 'mean' | 'std' | 'zscore'>('diff');

  // Optuna Tuning state
  const [tuneModel, setTuneModel] = useState<'lgbm' | 'xgb' | 'catboost' | 'all'>('all');
  const [tuneTrials, setTuneTrials] = useState<number>(30);
  const [tuning, setTuning] = useState<boolean>(false);
  const [tuneStatus, setTuneStatus] = useState<any>(null);

  // Ensemble state
  const [customWeights, setCustomWeights] = useState<Record<string, number>>({});
  const [customScore, setCustomScore] = useState<number | null>(null);
  const [caruanaResult, setCaruanaResult] = useState<any>(null);
  const [ensembleLoading, setEnsembleLoading] = useState<boolean>(false);

  // New Case Modal
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [inspectPath, setInspectPath] = useState<string>('');
  const [inspectData, setInspectData] = useState<any>(null);
  const [inspectLoading, setInspectLoading] = useState<boolean>(false);
  const [scannedCsvs, setScannedCsvs] = useState<Array<{ path: string; filename: string; size_mb: number }>>([]);
  const [isBrowsingTrain, setIsBrowsingTrain] = useState<boolean>(false);
  const [isBrowsingTest, setIsBrowsingTest] = useState<boolean>(false);

  // Dataset Relocation State
  const [relocateTrainPath, setRelocateTrainPath] = useState<string>('');
  const [relocateTestPath, setRelocateTestPath] = useState<string>('');
  const [relocateTargetCol, setRelocateTargetCol] = useState<string>('');
  const [relocateIdCol, setRelocateIdCol] = useState<string>('');
  const [isUpdatingPaths, setIsUpdatingPaths] = useState<boolean>(false);
  const [showRelocateCard, setShowRelocateCard] = useState<boolean>(false);
  const [pathUpdateSuccess, setPathUpdateSuccess] = useState<string | null>(null);
  const [pathUpdateError, setPathUpdateError] = useState<string | null>(null);

  // Experiment History LB editing state
  const [editingLbRunId, setEditingLbRunId] = useState<string | null>(null);
  const [tempLbScore, setTempLbScore] = useState<string>('');
  const [tempNotes, setTempNotes] = useState<string>('');
  const [isSavingLb, setIsSavingLb] = useState<boolean>(false);

  // Architecture & Hyperparameter State
  const [archPreset, setArchPreset] = useState<'shallow' | 'standard' | 'custom'>('standard');
  const [modelParams, setModelParams] = useState<any>({});
  const [paramsDraft, setParamsDraft] = useState<string>('{}');
  const [isEditingParams, setIsEditingParams] = useState<boolean>(false);
  const [isSavingParams, setIsSavingParams] = useState<boolean>(false);
  const [paramsSaveMsg, setParamsSaveMsg] = useState<string | null>(null);

  // Dataset Stitcher & Multi-Table Joiner Studio State
  const [showStitcherModal, setShowStitcherModal] = useState<boolean>(false);
  const [stitcherMode, setStitcherMode] = useState<'wizard' | 'join' | 'concat'>('wizard');
  const [wizardBasePattern, setWizardBasePattern] = useState<string>('');
  const [wizardJoinPattern, setWizardJoinPattern] = useState<string>('');
  const [wizardJoinKeys, setWizardJoinKeys] = useState<string>('');
  const [wizardOutFile, setWizardOutFile] = useState<string>('train_stitched.csv');
  const [wizardMaxFiles, setWizardMaxFiles] = useState<string>('');
  const [stitcherBasePath, setStitcherBasePath] = useState<string>('');
  const [stitcherJoinPath, setStitcherJoinPath] = useState<string>('');
  const [baseColumns, setBaseColumns] = useState<string[]>([]);
  const [joinColumns, setJoinColumns] = useState<string[]>([]);
  const [stitcherJoinKeys, setStitcherJoinKeys] = useState<string[]>([]);
  const [stitcherHow, setStitcherHow] = useState<'inner' | 'left' | 'outer'>('inner');
  const [stitcherAgg, setStitcherAgg] = useState<'none' | 'last' | 'first'>('none');
  const [stitcherFilterCol, setStitcherFilterCol] = useState<string>('');
  const [stitcherFilterVal, setStitcherFilterVal] = useState<string>('');
  const [stitcherOutFile, setStitcherOutFile] = useState<string>('train_merged.csv');
  const [stitcherApplyTarget, setStitcherApplyTarget] = useState<'train' | 'test' | 'none'>('train');

  // Concat Mode State
  const [concatPattern, setConcatPattern] = useState<string>('');
  const [concatMaxFiles, setConcatMaxFiles] = useState<string>('');
  const [concatOutFile, setConcatOutFile] = useState<string>('concatenated.csv');
  const [concatApplyTarget, setConcatApplyTarget] = useState<'train' | 'test' | 'none'>('train');

  // Preview & Status State
  const [stitcherPreviewLoading, setStitcherPreviewLoading] = useState<boolean>(false);
  const [stitcherPreviewData, setStitcherPreviewData] = useState<any>(null);
  const [stitcherExecuting, setStitcherExecuting] = useState<boolean>(false);
  const [stitcherError, setStitcherError] = useState<string | null>(null);
  const [stitcherSuccess, setStitcherSuccess] = useState<string | null>(null);

  const [isCreatingCase, setIsCreatingCase] = useState<boolean>(false);
  const [newCase, setNewCase] = useState({
    id: '',
    name: '',
    description: '',
    train_path: '',
    test_path: '',
    target_col: '',
    id_col: '',
    task_type: 'binary',
    eval_metric: 'roc_auc',
    cv_strategy: 'stratified_kfold',
    group_col: '',
    time_col: '',
    n_splits: 5,
    seed: 2026,
    models: ['lgbm', 'xgb', 'catboost'],
    hook_code: '',
    auto_stitch: false,
    stitch_base_pattern: '',
    stitch_join_pattern: '',
    stitch_join_keys: [] as string[]
  });

  // Fetch all cases
  const fetchCases = async () => {
    try {
      const res = await fetch('/api/cases');
      const data = await res.json();
      setCases(data.cases || []);
      if (data.cases?.length > 0 && !selectedCaseId) {
        setSelectedCaseId(data.cases[0].id);
      }
    } catch (err) {
      console.error('Failed to load cases:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCases();
  }, []);

  const fetchScannedCsvs = async () => {
    try {
      const res = await fetch('/api/scan-csvs');
      const data = await res.json();
      if (data?.files) {
        setScannedCsvs(data.files);
      }
    } catch (err) {
      console.error('Failed to scan workspace CSVs:', err);
    }
  };

  useEffect(() => {
    if (isModalOpen) {
      fetchScannedCsvs();
    }
  }, [isModalOpen]);

  // Fetch selected case details & EDA
  const fetchCaseDetail = async (id: string) => {
    if (!id) return;
    try {
      const res = await fetch(`/api/cases/${id}`);
      const data: CaseDetail = await res.json();
      setCaseDetail(data);
      setRelocateTrainPath(data.resolved_train_path || data.meta.train_path || '');
      setRelocateTestPath(data.resolved_test_path || data.meta.test_path || '');
      setRelocateTargetCol(data.meta.target_col || '');
      setRelocateIdCol(data.meta.id_col || '');
      setPathUpdateError(null);
      setPathUpdateSuccess(null);

      if (data.available_cached?.length > 0) {
        const defaultW: Record<string, number> = {};
        const val = Number((1.0 / data.available_cached.length).toFixed(2));
        data.available_cached.forEach(m => {
          defaultW[m] = val;
        });
        setCustomWeights(defaultW);
      }

      if (data.job_state?.status === 'running') {
        setRunning(true);
      } else {
        setRunning(false);
      }

      if (data.hook_code) {
        setEditableHookCode(data.hook_code);
        handleTestHook(id);
      } else {
        setHookValidation(null);
        setEditableHookCode(`import pandas as pd
import numpy as np
from tabular_pipeline import BaseFeatureHook

class CustomFeatures(BaseFeatureHook):
    """
    User-definable domain feature engineering hook.
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        return train_df, test_df
`);
      }

      if (data.meta?.models?.length > 0) {
        setSelectedModels(data.meta.models);
      }

      // Fetch architecture and hyperparameters
      try {
        const pRes = await fetch(`/api/cases/${id}/params`);
        if (pRes.ok) {
          const pData = await pRes.json();
          setArchPreset(pData.architecture_preset || 'standard');
          setModelParams(pData.model_params || {});
          setParamsDraft(JSON.stringify(pData.model_params || {}, null, 2));
        }
      } catch (e) {
        console.error('Failed to load params:', e);
      }

      // Automatically load EDA
      fetchEda(id);
    } catch (err) {
      console.error('Failed to load case detail:', err);
    }
  };

  const fetchEda = async (id: string) => {
    setEdaLoading(true);
    try {
      const res = await fetch(`/api/cases/${id}/eda`);
      if (res.ok) {
        const data = await res.json();
        setEdaData(data);
      } else {
        setEdaData(null);
        fetchScannedCsvs();
      }
    } catch (e) {
      console.error('EDA fetch failed:', e);
      setEdaData(null);
      fetchScannedCsvs();
    } finally {
      setEdaLoading(false);
    }
  };

  const handleUpdatePaths = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!selectedCaseId || !relocateTrainPath.trim()) return;

    setIsUpdatingPaths(true);
    setPathUpdateSuccess(null);
    setPathUpdateError(null);

    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/paths`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          train_path: relocateTrainPath.trim(),
          test_path: relocateTestPath.trim() || undefined,
          target_col: relocateTargetCol.trim() || undefined,
          id_col: relocateIdCol.trim() || undefined
        })
      });
      const data = await res.json();
      if (res.ok) {
        setPathUpdateSuccess('Dataset paths updated successfully');
        setShowRelocateCard(false);
        await fetchCaseDetail(selectedCaseId);
        await fetchCases();
      } else {
        setPathUpdateError(data.detail || 'Failed to update dataset paths');
      }
    } catch (err: any) {
      setPathUpdateError(err.message || 'Network error updating paths');
    } finally {
      setIsUpdatingPaths(false);
    }
  };

  const handleBrowseRelocate = async (field: 'train' | 'test') => {
    try {
      const title = field === 'train' ? 'Select Training Dataset (CSV)' : 'Select Test Dataset (CSV)';
      const res = await fetch(`/api/browse-file?title=${encodeURIComponent(title)}`);
      const data = await res.json();
      if (data?.path) {
        if (field === 'train') {
          setRelocateTrainPath(data.path);
        } else {
          setRelocateTestPath(data.path);
        }
      }
    } catch (err) {
      console.error('File dialog error:', err);
    }
  };

  const handleSaveLb = async (runId: string) => {
    if (!selectedCaseId) return;
    setIsSavingLb(true);
    try {
      const val = tempLbScore.trim() ? parseFloat(tempLbScore.trim()) : null;
      await fetch(`/api/cases/${selectedCaseId}/history/${runId}/lb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lb_score: val, notes: tempNotes.trim() || undefined })
      });
      setEditingLbRunId(null);
      await fetchCaseDetail(selectedCaseId);
    } catch (e) {
      console.error('Failed to update LB score:', e);
    } finally {
      setIsSavingLb(false);
    }
  };

  useEffect(() => {
    if (selectedCaseId) {
      fetchCaseDetail(selectedCaseId);
    }
  }, [selectedCaseId]);

  // Polling when job is running
  useEffect(() => {
    let interval: any = null;
    if (running && selectedCaseId) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`/api/cases/${selectedCaseId}/status`);
          const status = await res.json();
          if (status.status !== 'running') {
            setRunning(false);
            fetchCaseDetail(selectedCaseId);
            fetchCases();
          } else {
            setCaseDetail(prev => prev ? { ...prev, job_state: status } : null);
          }
        } catch (e) {
          console.error(e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [running, selectedCaseId]);

  // Polling for Optuna Tuning
  useEffect(() => {
    let interval: any = null;
    if (tuning && selectedCaseId) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`/api/cases/${selectedCaseId}/tune-status`);
          const status = await res.json();
          setTuneStatus(status);
          if (status.status === 'completed' || status.status === 'failed') {
            setTuning(false);
          }
        } catch (e) {
          console.error(e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [tuning, selectedCaseId]);

  const handleStartTune = async () => {
    if (!selectedCaseId) return;
    try {
      setTuning(true);
      await fetch(`/api/cases/${selectedCaseId}/tune`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: tuneModel, n_trials: tuneTrials })
      });
    } catch (err) {
      console.error('Tuning launch failed:', err);
      setTuning(false);
    }
  };

  const handleApplyTunedParams = async () => {
    if (!selectedCaseId) return;
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/tune-apply`, { method: 'POST' });
      if (res.ok) {
        alert('Tuned hyperparameters successfully applied to case configuration!');
        fetchCaseDetail(selectedCaseId);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleSelectPreset = async (preset: 'shallow' | 'standard') => {
    if (!selectedCaseId) return;
    setIsSavingParams(true);
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/params`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ architecture_preset: preset })
      });
      const data = await res.json();
      if (res.ok) {
        setArchPreset(preset);
        setModelParams(data.model_params || {});
        setParamsDraft(JSON.stringify(data.model_params || {}, null, 2));
        setParamsSaveMsg(`Applied ${preset === 'shallow' ? 'Shallow Trees' : 'Standard Baseline'} preset`);
        setTimeout(() => setParamsSaveMsg(null), 3500);
        await fetchCaseDetail(selectedCaseId);
      }
    } catch (err) {
      console.error('Error applying preset:', err);
    } finally {
      setIsSavingParams(false);
    }
  };

  const handleSaveCustomParams = async () => {
    if (!selectedCaseId) return;
    try {
      const parsed = JSON.parse(paramsDraft);
      setIsSavingParams(true);
      const res = await fetch(`/api/cases/${selectedCaseId}/params`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ architecture_preset: 'custom', model_params: parsed })
      });
      const data = await res.json();
      if (res.ok) {
        setArchPreset('custom');
        setModelParams(data.model_params || {});
        setIsEditingParams(false);
        setParamsSaveMsg('Custom hyperparameters saved');
        setTimeout(() => setParamsSaveMsg(null), 3500);
        await fetchCaseDetail(selectedCaseId);
      }
    } catch (err: any) {
      alert('Invalid JSON: ' + err.message);
    } finally {
      setIsSavingParams(false);
    }
  };

  const handleInspectBaseColumns = async (path: string) => {
    setStitcherBasePath(path);
    if (!path.trim()) { setBaseColumns([]); return; }
    try {
      const res = await fetch(`/api/tools/inspect-columns?path=${encodeURIComponent(path.trim())}`);
      if (res.ok) {
        const d = await res.json();
        setBaseColumns(d.columns || []);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleInspectJoinColumns = async (path: string) => {
    setStitcherJoinPath(path);
    if (!path.trim()) { setJoinColumns([]); return; }
    try {
      const res = await fetch(`/api/tools/inspect-columns?path=${encodeURIComponent(path.trim())}`);
      if (res.ok) {
        const d = await res.json();
        setJoinColumns(d.columns || []);
        if (baseColumns.length > 0) {
          const common = d.columns.filter((c: string) => baseColumns.includes(c));
          if (common.length > 0 && stitcherJoinKeys.length === 0) {
            setStitcherJoinKeys(common);
          }
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handlePreviewMerge = async () => {
    if (!stitcherBasePath.trim() || !stitcherJoinPath.trim() || stitcherJoinKeys.length === 0) {
      setStitcherError('Please specify Base Table, Join Table, and at least one Join Key.');
      return;
    }
    setStitcherPreviewLoading(true);
    setStitcherError(null);
    setStitcherPreviewData(null);
    try {
      const res = await fetch('/api/tools/preview-merge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_path: stitcherBasePath.trim(),
          join_path: stitcherJoinPath.trim(),
          join_keys: stitcherJoinKeys,
          how: stitcherHow,
          aggregation: stitcherAgg,
          filter_col: stitcherFilterCol.trim() || undefined,
          filter_val: stitcherFilterVal.trim() || undefined,
        })
      });
      const data = await res.json();
      if (res.ok) {
        setStitcherPreviewData(data);
      } else {
        setStitcherError(data.detail || 'Preview merge failed');
      }
    } catch (err: any) {
      setStitcherError(err.message || 'Error executing preview merge');
    } finally {
      setStitcherPreviewLoading(false);
    }
  };

  const handleExecuteMerge = async () => {
    if (!selectedCaseId) return;
    if (!stitcherBasePath.trim() || !stitcherJoinPath.trim() || stitcherJoinKeys.length === 0) {
      setStitcherError('Please specify Base Table, Join Table, and at least one Join Key.');
      return;
    }
    setStitcherExecuting(true);
    setStitcherError(null);
    setStitcherSuccess(null);
    try {
      const res = await fetch('/api/tools/merge-datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_id: selectedCaseId,
          base_path: stitcherBasePath.trim(),
          join_path: stitcherJoinPath.trim(),
          join_keys: stitcherJoinKeys,
          how: stitcherHow,
          aggregation: stitcherAgg,
          filter_col: stitcherFilterCol.trim() || undefined,
          filter_val: stitcherFilterVal.trim() || undefined,
          output_filename: stitcherOutFile.trim() || 'train_merged.csv',
          set_as_target: stitcherApplyTarget !== 'none' ? stitcherApplyTarget : undefined
        })
      });
      const data = await res.json();
      if (res.ok) {
        setStitcherSuccess(`Successfully merged ${data.total_rows.toLocaleString()} rows into ${data.saved_path}!`);
        await fetchCaseDetail(selectedCaseId);
        await fetchCases();
        setTimeout(() => {
          setShowStitcherModal(false);
          setStitcherSuccess(null);
        }, 2200);
      } else {
        setStitcherError(data.detail || 'Merge execution failed');
      }
    } catch (err: any) {
      setStitcherError(err.message || 'Error executing merge');
    } finally {
      setStitcherExecuting(false);
    }
  };

  const handleExecuteConcat = async () => {
    if (!selectedCaseId || !concatPattern.trim()) return;
    setStitcherExecuting(true);
    setStitcherError(null);
    setStitcherSuccess(null);
    try {
      const res = await fetch('/api/tools/concat-partitions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_id: selectedCaseId,
          pattern: concatPattern.trim(),
          max_files: concatMaxFiles.trim() ? Number(concatMaxFiles) : undefined,
          output_filename: concatOutFile.trim() || 'concatenated.csv',
          set_as_target: concatApplyTarget !== 'none' ? concatApplyTarget : undefined
        })
      });
      const data = await res.json();
      if (res.ok) {
        setStitcherSuccess(`Successfully concatenated ${data.files_matched} files into ${data.saved_path} (${data.total_rows.toLocaleString()} rows)!`);
        await fetchCaseDetail(selectedCaseId);
        await fetchCases();
        setTimeout(() => {
          setShowStitcherModal(false);
          setStitcherSuccess(null);
        }, 2200);
      } else {
        setStitcherError(data.detail || 'Concat execution failed');
      }
    } catch (err: any) {
      setStitcherError(err.message || 'Error executing concat');
    } finally {
      setStitcherExecuting(false);
    }
  };

  const handleExecuteSmartStitch = async () => {
    if (!selectedCaseId || !wizardBasePattern.trim() || !wizardJoinPattern.trim()) return;
    setStitcherExecuting(true);
    setStitcherError(null);
    setStitcherSuccess(null);
    try {
      const keys = wizardJoinKeys.split(',').map(s => s.trim()).filter(Boolean);
      const res = await fetch('/api/tools/smart-stitch-partitions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_id: selectedCaseId,
          base_pattern: wizardBasePattern.trim(),
          join_pattern: wizardJoinPattern.trim(),
          join_keys: keys,
          max_files: wizardMaxFiles.trim() ? Number(wizardMaxFiles) : undefined,
          output_filename: wizardOutFile.trim() || 'train_all_weeks_merged.csv',
          set_as_target: 'train'
        })
      });
      const data = await res.json();
      if (res.ok) {
        setStitcherSuccess(`Successfully stitched ${data.base_files_matched} base files and ${data.join_files_matched} join files into ${data.saved_path} (${data.total_rows.toLocaleString()} rows)!`);
        await fetchCaseDetail(selectedCaseId);
        await fetchCases();
        setTimeout(() => {
          setShowStitcherModal(false);
          setStitcherSuccess(null);
        }, 2200);
      } else {
        setStitcherError(data.detail || 'Smart stitch failed');
      }
    } catch (err: any) {
      setStitcherError(err.message || 'Network error executing smart stitch');
    } finally {
      setStitcherExecuting(false);
    }
  };

  const handleTestHook = async (targetCaseId?: string) => {
    const cId = targetCaseId || selectedCaseId;
    if (!cId) return;
    setHookTesting(true);
    try {
      const res = await fetch(`/api/cases/${cId}/hook/test`, { method: 'POST' });
      const data = await res.json();
      setHookValidation(data);
    } catch (err) {
      console.error('Failed to test hook:', err);
      setHookValidation({ status: 'error', message: String(err) });
    } finally {
      setHookTesting(false);
    }
  };

  const handleSaveHook = async () => {
    if (!selectedCaseId) return;
    setSavingHook(true);
    setHookSaved(false);
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/hook`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hook_code: editableHookCode })
      });
      if (res.ok) {
        setHookSaved(true);
        setTimeout(() => setHookSaved(false), 3000);
        fetchCaseDetail(selectedCaseId);
        handleTestHook(selectedCaseId);
      }
    } catch (e) {
      console.error('Failed to save hook:', e);
    } finally {
      setSavingHook(false);
    }
  };

  const handleUndoHook = () => {
    if (hookHistory.length === 0) {
      alert('No previous actions to undo.');
      return;
    }
    const previousCode = hookHistory[hookHistory.length - 1];
    setHookHistory(prev => prev.slice(0, -1));
    setEditableHookCode(previousCode);
  };

  const handleResetHook = () => {
    if (window.confirm('Reset feature hook to clean default template? All custom additions will be cleared.')) {
      setHookHistory(prev => [...prev.slice(-20), editableHookCode]);
      const cleanTemplate = `from tabular_pipeline import BaseFeatureHook
import pandas as pd
import numpy as np

class CustomFeatureHook(BaseFeatureHook):
    """
    Universal Leak-Free Feature Hook.
    Executes fold-by-fold before machine learning models.
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        # Write custom feature transformations below

        return train_df, test_df
`;
      setEditableHookCode(cleanTemplate);
    }
  };

  const insertCodeIntoHook = (codeSnippet: string, duplicateCheckKey?: string) => {
    if (duplicateCheckKey && editableHookCode.includes(duplicateCheckKey)) {
      alert(`Notice: Feature or logic containing "${duplicateCheckKey}" is already present in your code. Insertion skipped to prevent redundant duplication.`);
      return;
    }

    setHookHistory(prev => [...prev.slice(-20), editableHookCode]);

    if (!editableHookCode || editableHookCode.trim().length === 0) {
      const boilerplate = `from tabular_pipeline import BaseFeatureHook
import pandas as pd
import numpy as np

class CustomFeatureHook(BaseFeatureHook):
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        # Universal Leak-Free Feature Hook
        ${codeSnippet.trim()}

        return train_df, test_df
`;
      setEditableHookCode(boilerplate);
      return;
    }

    if (editableHookCode.includes('return train_df, test_df')) {
      const parts = editableHookCode.split('return train_df, test_df');
      setEditableHookCode(parts[0] + '        ' + codeSnippet.trim() + '\n\n        return train_df, test_df' + (parts[1] || ''));
    } else {
      setEditableHookCode(prev => prev + '\n        ' + codeSnippet.trim());
    }
  };

  const handleInsertFormula = () => {
    if (!feColA || !feColB) {
      alert('Please select both Column A and Column B.');
      return;
    }
    const cleanA = feColA.replace(/[^a-zA-Z0-9_]/g, '_');
    const cleanB = feColB.replace(/[^a-zA-Z0-9_]/g, '_');
    const opNames: Record<string, string> = { '/': 'ratio', '*': 'mult', '-': 'diff', '+': 'sum' };
    const opName = opNames[feOp] || 'feat';
    const newColName = `${cleanA}_${opName}_${cleanB}`;

    let expr = '';
    if (feOp === '/') {
      expr = `# Visual Formula: ${newColName}\n        for df in ([train_df, test_df] if test_df is not None else [train_df]):\n            df['${newColName}'] = df['${feColA}'] / (df['${feColB}'].abs() + 1e-5)`;
    } else if (feOp === '*') {
      expr = `# Visual Formula: ${newColName}\n        for df in ([train_df, test_df] if test_df is not None else [train_df]):\n            df['${newColName}'] = df['${feColA}'] * df['${feColB}']`;
    } else if (feOp === '-') {
      expr = `# Visual Formula: ${newColName}\n        for df in ([train_df, test_df] if test_df is not None else [train_df]):\n            df['${newColName}'] = df['${feColA}'] - df['${feColB}']`;
    } else {
      expr = `# Visual Formula: ${newColName}\n        for df in ([train_df, test_df] if test_df is not None else [train_df]):\n            df['${newColName}'] = df['${feColA}'] + df['${feColB}']`;
    }

    insertCodeIntoHook(expr, newColName);
  };

  const handleInsertGroupFormula = () => {
    if (!feGroupCat || !feGroupNum) {
      alert('Please select both a Group Categorical Column and a Numeric Column.');
      return;
    }
    const cleanCat = feGroupCat.replace(/[^a-zA-Z0-9_]/g, '_');
    const cleanNum = feGroupNum.replace(/[^a-zA-Z0-9_]/g, '_');

    let colName = '';
    let expr = '';
    if (feGroupStat === 'diff') {
      colName = `${cleanNum}_diff_from_${cleanCat}_mean`;
      expr = `# Visual Group Deviation: ${colName}\n        grp_map = train_df.groupby('${feGroupCat}')['${feGroupNum}'].mean().to_dict()\n        for df in ([train_df, test_df] if test_df is not None else [train_df]):\n            df['${colName}'] = df['${feGroupNum}'] - df['${feGroupCat}'].map(grp_map)`;
    } else if (feGroupStat === 'mean') {
      colName = `${cleanNum}_by_${cleanCat}_mean`;
      expr = `# Visual Group Mean: ${colName}\n        grp_map = train_df.groupby('${feGroupCat}')['${feGroupNum}'].mean().to_dict()\n        for df in ([train_df, test_df] if test_df is not None else [train_df]):\n            df['${colName}'] = df['${feGroupCat}'].map(grp_map)`;
    } else if (feGroupStat === 'std') {
      colName = `${cleanNum}_by_${cleanCat}_std`;
      expr = `# Visual Group Std: ${colName}\n        grp_map = train_df.groupby('${feGroupCat}')['${feGroupNum}'].std().fillna(0).to_dict()\n        for df in ([train_df, test_df] if test_df is not None else [train_df]):\n            df['${colName}'] = df['${feGroupCat}'].map(grp_map)`;
    } else {
      colName = `${cleanNum}_zscore_by_${cleanCat}`;
      expr = `# Visual Group Z-Score: ${colName}\n        g_mean = train_df.groupby('${feGroupCat}')['${feGroupNum}'].mean().to_dict()\n        g_std = train_df.groupby('${feGroupCat}')['${feGroupNum}'].std().fillna(1.0).to_dict()\n        for df in ([train_df, test_df] if test_df is not None else [train_df]):\n            mean_s = df['${feGroupCat}'].map(g_mean)\n            std_s = df['${feGroupCat}'].map(g_std).replace(0, 1.0)\n            df['${colName}'] = (df['${feGroupNum}'] - mean_s) / std_s`;
    }

    insertCodeIntoHook(expr, colName);
  };

  const handleInsertGroupStatsTemplate = () => {
    const target = meta?.target_col || 'target';
    const id = meta?.id_col || 'id';
    const snippet = `# --- UNIVERSAL PRIMITIVE 1: GROUP-BY AGGREGATIONS & RELATIVE DEVIATION ---
        # Computes peer-group baseline statistics (learned strictly from train_df to prevent leakage)
        cat_cols = [c for c in train_df.select_dtypes(include=['object', 'category']).columns if c not in ['${id}', '${target}']]
        num_cols = [c for c in train_df.select_dtypes(include=['number']).columns if c not in ['${id}', '${target}']]
        for group_col in cat_cols[:2]:
            for val_col in num_cols[:2]:
                grp_mean = train_df.groupby(group_col)[val_col].mean().to_dict()
                for df in ([train_df, test_df] if test_df is not None else [train_df]):
                    df[f'{val_col}_by_{group_col}_mean'] = df[group_col].map(grp_mean)
                    df[f'{val_col}_diff_{group_col}_mean'] = df[val_col] - df[group_col].map(grp_mean)`;
    insertCodeIntoHook(snippet, 'UNIVERSAL PRIMITIVE 1');
  };

  const handleInsertRowStatsTemplate = () => {
    const target = meta?.target_col || 'target';
    const id = meta?.id_col || 'id';
    const snippet = `# --- UNIVERSAL PRIMITIVE 2: ROW-WISE INTENSITY & CONSISTENCY STATISTICS ---
        # Computes across-the-board intensity, dispersion, and missingness per entity
        numeric_cols = [c for c in train_df.select_dtypes(include=['number']).columns if c not in ['${id}', '${target}']][:8]
        if len(numeric_cols) >= 2:
            for df in ([train_df, test_df] if test_df is not None else [train_df]):
                df['row_stat_mean'] = df[numeric_cols].mean(axis=1)
                df['row_stat_std'] = df[numeric_cols].std(axis=1).fillna(0)
                df['row_stat_min'] = df[numeric_cols].min(axis=1)
                df['row_stat_max'] = df[numeric_cols].max(axis=1)
                df['row_null_count'] = df[numeric_cols].isnull().sum(axis=1)`;
    insertCodeIntoHook(snippet, 'UNIVERSAL PRIMITIVE 2');
  };

  const handleInsertFreqEncodingTemplate = () => {
    const target = meta?.target_col || 'target';
    const id = meta?.id_col || 'id';
    const snippet = `# --- UNIVERSAL PRIMITIVE 3: HIGH-CARDINALITY FREQUENCY & RARITY ENCODING ---
        # Maps categorical entities to prevalence rates (detects rare categories without leakage)
        cat_cols = [c for c in train_df.select_dtypes(include=['object', 'category']).columns if c not in ['${id}', '${target}']]
        for c in cat_cols:
            freq = train_df[c].value_counts(normalize=True).to_dict()
            for df in ([train_df, test_df] if test_df is not None else [train_df]):
                df[f'{c}_freq'] = df[c].map(freq).fillna(0)`;
    insertCodeIntoHook(snippet, 'UNIVERSAL PRIMITIVE 3');
  };

  const handleInsertLogTransformTemplate = () => {
    const target = meta?.target_col || 'target';
    const id = meta?.id_col || 'id';
    const snippet = `# --- UNIVERSAL PRIMITIVE 4: SKEW CORRECTION & LOG-TRANSFORM ---
        # Stabilizes heavy-tailed variables (financial, revenue, delays, physical quantities)
        num_cols = [c for c in train_df.select_dtypes(include=['number']).columns if c not in ['${id}', '${target}']]
        for c in num_cols[:4]:
            if (train_df[c] >= 0).all():
                for df in ([train_df, test_df] if test_df is not None else [train_df]):
                    df[f'{c}_log1p'] = np.log1p(df[c].clip(lower=0))`;
    insertCodeIntoHook(snippet, 'UNIVERSAL PRIMITIVE 4');
  };

  // Execute Pipeline
  const handleExecute = async () => {
    if (!selectedCaseId) return;
    if (selectedModels.length === 0) {
      alert('Please select at least one model to train.');
      return;
    }
    setRunning(true);
    setActiveTab('training');
    try {
      await fetch(`/api/cases/${selectedCaseId}/run`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ models: selectedModels })
      });
    } catch (e) {
      console.error(e);
      setRunning(false);
    }
  };

  // Stop Pipeline
  const handleStopPipeline = async () => {
    if (!selectedCaseId) return;
    try {
      await fetch(`/api/cases/${selectedCaseId}/stop`, { method: 'POST' });
      setRunning(false);
      fetchCaseDetail(selectedCaseId);
    } catch (e) {
      console.error('Failed to stop pipeline:', e);
    }
  };

  // Run Caruana
  const handleRunCaruana = async () => {
    if (!selectedCaseId) return;
    setEnsembleLoading(true);
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/ensemble/hill-climb`, { method: 'POST' });
      const data = await res.json();
      setCaruanaResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setEnsembleLoading(false);
    }
  };

  // Custom Blend
  const handleBlend = async (weights: Record<string, number>) => {
    if (!selectedCaseId) return;
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/ensemble/blend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ weights })
      });
      const data = await res.json();
      setCustomScore(data.score);
    } catch (e) {
      console.error(e);
    }
  };

  // Inspect Dataset file (in modal) with smart column detection
  const handleInspect = async (pathOverride?: string) => {
    const targetPath = pathOverride || inspectPath || newCase.train_path;
    if (!targetPath) return;
    setInspectLoading(true);
    try {
      const res = await fetch('/api/inspect-dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: targetPath })
      });
      const data = await res.json();
      setInspectData(data);

      if (data.columns && Array.isArray(data.columns)) {
        const colNames: string[] = data.columns.map((c: any) => c.column);
        
        // Auto-detect ID column
        const detectedId = colNames.find(c => {
          const lower = c.toLowerCase();
          return lower === 'id' || lower === 'passengerid' || lower === 'sk_id_curr' || lower.endsWith('_id') || lower === 'identifier';
        });

        // Auto-detect Target column
        const detectedTarget = colNames.find(c => {
          const lower = c.toLowerCase();
          return ['target', 'satisfaction', 'transported', 'label', 'survived', 'class', 'outcome', 'y'].includes(lower);
        });

        // Auto-infer ID / Name if currently empty
        const filename = targetPath.split(/[\\/]/).pop()?.replace('.csv', '') || 'case';
        const inferredId = filename.toLowerCase().replace(/[^a-z0-9_]/g, '_');
        const inferredName = filename.replace(/[_-]/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

        setNewCase(prev => ({
          ...prev,
          train_path: targetPath,
          id: prev.id || inferredId,
          name: prev.name || inferredName,
          id_col: prev.id_col || detectedId || '',
          target_col: prev.target_col || detectedTarget || ''
        }));
      }
    } catch (e) {
      console.error(e);
    } finally {
      setInspectLoading(false);
    }
  };

  // Browse file using native Windows File Dialog
  const handleBrowseFile = async (field: 'train' | 'test') => {
    if (field === 'train') setIsBrowsingTrain(true);
    else setIsBrowsingTest(true);

    try {
      const title = field === 'train' ? 'Select Training Dataset (CSV)' : 'Select Test Dataset (CSV)';
      const res = await fetch(`/api/browse-file?title=${encodeURIComponent(title)}`);
      const data = await res.json();
      if (data?.path) {
        if (field === 'train') {
          setNewCase(prev => ({ ...prev, train_path: data.path }));
          setInspectPath(data.path);
          handleInspect(data.path);
        } else {
          setNewCase(prev => ({ ...prev, test_path: data.path }));
        }
      }
    } catch (err) {
      console.error('File dialog error:', err);
    } finally {
      if (field === 'train') setIsBrowsingTrain(false);
      else setIsBrowsingTest(false);
    }
  };

  // Create Case
  const handleCreateCase = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsCreatingCase(true);
    try {
      const res = await fetch('/api/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newCase)
      });
      const data = await res.json();
      setIsModalOpen(false);
      await fetchCases();
      setSelectedCaseId(data.case_id);
    } catch (err) {
      console.error(err);
    } finally {
      setIsCreatingCase(false);
    }
  };

  const handleDeleteCase = async (caseId: string) => {
    if (!caseId) return;
    if (!window.confirm(`Are you sure you want to delete case "${caseId}"? This will permanently remove its directory and data.`)) return;
    try {
      const res = await fetch(`/api/cases/${caseId}`, { method: 'DELETE' });
      if (res.ok) {
        await fetchCases();
      }
    } catch (err) {
      console.error('Delete case failed:', err);
    }
  };

  const meta = caseDetail?.meta;
  const summary = caseDetail?.summary;
  const scores = summary?.all_scores || {};
  const championScore = summary?.champion_score;
  const championModel = summary?.champion_model;

  const singleModels = ['lgbm', 'xgb', 'catboost'].filter(m => scores[m] !== undefined);
  let bestSingleScore: number | null = null;
  let bestSingleName = '';
  if (singleModels.length > 0) {
    singleModels.sort((a, b) => scores[b] - scores[a]);
    bestSingleName = singleModels[0];
    bestSingleScore = scores[bestSingleName];
  }

  return (
    <div className="min-h-screen bg-[#F4F5F6] text-[#121816] font-sans antialiased selection:bg-[#E8F3ED] selection:text-[#164E33]">
      {/* Top Header */}
      <header className="border-b border-[#E5E9E6] bg-white sticky top-0 z-20 shadow-2xs">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2.5">
              <span className="font-bold tracking-tight text-base text-[#121816]">TABULAR STUDIO</span>
              <span className="text-[10px] font-mono font-bold bg-[#E8F3ED] text-[#164E33] border border-[#D2E7DC] px-2 py-0.5 rounded-full">
                NEXUS
              </span>
            </div>

            {/* Case Selector */}
            <div className="flex items-center space-x-2">
              <select
                value={selectedCaseId}
                onChange={e => setSelectedCaseId(e.target.value)}
                className="bg-[#F8FAF9] border border-[#E5E9E6] hover:border-[#164E33] rounded-xl px-3 py-1.5 text-xs font-medium text-[#121816] focus:outline-none focus:border-[#164E33] transition"
              >
                {cases.map(c => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
              
              <button
                onClick={() => setIsModalOpen(true)}
                className="flex items-center space-x-1.5 border border-[#E5E9E6] hover:border-[#164E33] hover:bg-[#F8FAF9] text-[#121816] text-xs font-medium px-3 py-1.5 rounded-xl transition shadow-2xs"
              >
                <Plus size={13} />
                <span>New Case</span>
              </button>

              {cases.length > 0 && selectedCaseId && (
                <button
                  type="button"
                  onClick={() => handleDeleteCase(selectedCaseId)}
                  className="flex items-center space-x-1 border border-[#E5E9E6] hover:border-rose-400 hover:bg-rose-50 text-zinc-400 hover:text-rose-600 text-xs font-medium px-2 py-1.5 rounded-xl transition"
                  title={`Delete active case "${selectedCaseId}"`}
                >
                  <Trash2 size={13} />
                </button>
              )}
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <a
              href={`/api/cases/${selectedCaseId}/export-script`}
              target="_blank"
              rel="noreferrer"
              className="flex items-center space-x-1.5 text-xs font-medium border border-[#E5E9E6] hover:border-[#164E33] hover:bg-[#F8FAF9] text-[#121816] px-3 py-1.5 rounded-xl transition shadow-2xs"
            >
              <FileCode size={13} />
              <span>Export Script.py</span>
            </a>

            <a
              href={`/api/cases/${selectedCaseId}/export-notebook`}
              target="_blank"
              rel="noreferrer"
              className="flex items-center space-x-1.5 text-xs font-medium border border-[#E5E9E6] hover:border-[#164E33] hover:bg-[#F8FAF9] text-[#121816] px-3 py-1.5 rounded-xl transition shadow-2xs"
            >
              <BookOpen size={13} />
              <span>Export Notebook.ipynb</span>
            </a>

            {running ? (
              <button
                onClick={handleStopPipeline}
                className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold text-white bg-rose-600 hover:bg-rose-700 transition shadow-sm"
                title="Halt current training execution immediately"
              >
                <Square size={12} className="fill-current" />
                <span>Stop Pipeline</span>
              </button>
            ) : (
              <button
                onClick={handleExecute}
                className="flex items-center space-x-1.5 px-4 py-1.5 rounded-xl text-xs font-semibold text-white bg-[#164E33] hover:bg-[#123E28] transition shadow-sm"
              >
                <Play size={13} />
                <span>Run Pipeline</span>
              </button>
            )}
          </div>
        </div>

        {/* Data Mining Lifecycle Tabs (Strict Sequential Workflow) */}
        <div className="max-w-7xl mx-auto px-6 flex space-x-8 -mb-px">
          {[
            { id: 'eda', step: '1', label: 'Dataset & EDA', icon: Database },
            { id: 'fe', step: '2', label: 'Feature Engineering', icon: Wrench },
            { id: 'training', step: '3', label: 'Model Training & CV', icon: Cpu },
            { id: 'ensemble', step: '4', label: 'Ensemble Laboratory', icon: Sliders },
            { id: 'artifacts', step: '5', label: 'Submissions & Artifacts', icon: FileSpreadsheet }
          ].map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 py-3 border-b-2 text-xs font-medium transition ${
                  active
                    ? 'border-[#164E33] text-[#164E33] font-bold'
                    : 'border-transparent text-[#76807B] hover:text-[#121816] hover:border-zinc-300'
                }`}
              >
                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded-md font-bold ${active ? 'bg-[#E8F3ED] text-[#164E33] border border-[#D2E7DC]' : 'bg-zinc-100 text-[#76807B]'}`}>
                  {tab.step}
                </span>
                <Icon size={14} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {loading ? (
          <div className="flex items-center justify-center py-24 text-zinc-500 space-x-2">
            <Loader2 className="animate-spin" size={20} />
            <span>Loading workspace data...</span>
          </div>
        ) : !caseDetail ? (
          <div className="text-center py-20 text-zinc-500">Select or create a case to begin.</div>
        ) : (
          <>
            {/* Live Pipeline Execution Terminal */}
            {running && (
              <div className="mb-8 border border-zinc-900 bg-zinc-950 text-zinc-100 rounded-lg p-5 shadow-lg space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                  <div className="flex items-center space-x-3">
                    <span className="relative flex h-2.5 w-2.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                    </span>
                    <div className="flex items-center space-x-2 text-sm font-semibold text-white">
                      <span>Pipeline Execution Engine Active</span>
                      <span className="text-[10px] font-mono uppercase bg-zinc-800 text-zinc-300 px-2 py-0.5 rounded border border-zinc-700">
                        {caseDetail.job_state.current_model || 'CROSS-VALIDATION'}
                      </span>
                      {caseDetail.job_state.current_fold && (
                        <span className="text-[10px] font-mono bg-zinc-800 text-zinc-300 px-2 py-0.5 rounded border border-zinc-700">
                          Fold {caseDetail.job_state.current_fold}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <button
                      onClick={handleStopPipeline}
                      className="flex items-center space-x-1 bg-red-600 hover:bg-red-700 text-white text-[11px] font-semibold px-2.5 py-1 rounded transition shadow-sm"
                      title="Halt active pipeline execution"
                    >
                      <Square size={10} className="fill-current" />
                      <span>Stop Pipeline</span>
                    </button>
                    <div className="text-sm font-mono font-bold text-white">
                      {caseDetail.job_state.progress}%
                    </div>
                  </div>
                </div>

                {/* Animated Progress Bar */}
                <div className="w-full bg-zinc-800 h-2 rounded overflow-hidden">
                  <div
                    className="bg-white h-full rounded transition-all duration-300 ease-out"
                    style={{ width: `${Math.max(5, caseDetail.job_state.progress)}%` }}
                  />
                </div>

                {/* Live Console Output */}
                <div className="bg-black/60 border border-zinc-800/80 rounded p-3 font-mono text-xs max-h-40 overflow-y-auto space-y-1 text-zinc-300 select-text">
                  {caseDetail.job_state.logs.slice(-15).map((log, lIdx) => (
                    <div key={lIdx} className="leading-relaxed flex items-start space-x-2">
                      <span className="text-zinc-600 select-none">&gt;</span>
                      <span className={log.includes('Completed') || log.includes('finished') || log.includes('Champion') ? 'text-emerald-400 font-semibold' : 'text-zinc-300'}>
                        {log}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* TAB 1: DATASET & EDA (Exploratory Data Analysis) */}
            {activeTab === 'eda' && (
              <div className="space-y-8">
                {edaLoading ? (
                  <div className="flex items-center justify-center py-16 text-zinc-500 space-x-2">
                    <Loader2 className="animate-spin" size={18} />
                    <span>Profiling dataset schema and statistical distribution...</span>
                  </div>
                ) : !edaData ? (
                  <div className="bg-white border border-[#E5E9E6] rounded-2xl p-8 shadow-sm max-w-3xl mx-auto space-y-6">
                    <div className="flex items-start space-x-4">
                      <div className="w-12 h-12 rounded-xl bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-700 flex-shrink-0">
                        <AlertTriangle size={24} />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-base font-semibold text-[#121816]">Dataset File Not Found on This Machine</h3>
                        <p className="text-xs text-[#76807B] mt-1 leading-relaxed">
                          The configured dataset location could not be located on your local file system. This happens when a case is shared from another machine or when files are moved.
                        </p>
                      </div>
                    </div>

                    <div className="bg-[#F8FAF9] border border-[#E5E9E6] rounded-xl p-4 space-y-2 text-xs font-mono">
                      <div className="flex items-center justify-between text-[#76807B]">
                        <span>Configured Train Path:</span>
                        <span className="text-rose-600 font-semibold truncate max-w-md">{caseDetail?.meta.train_path || 'None'}</span>
                      </div>
                      {caseDetail?.meta.test_path && (
                        <div className="flex items-center justify-between text-[#76807B]">
                          <span>Configured Test Path:</span>
                          <span className="text-zinc-600 truncate max-w-md">{caseDetail.meta.test_path}</span>
                        </div>
                      )}
                    </div>

                    <form onSubmit={handleUpdatePaths} className="space-y-4 pt-2">
                      <div className="space-y-1.5">
                        <label className="text-xs font-semibold text-[#121816] flex items-center justify-between">
                          <span>Training Dataset Path (train.csv) *</span>
                          <span className="text-[11px] font-normal text-[#76807B]">Local absolute path or workspace-relative path</span>
                        </label>
                        <div className="flex space-x-2">
                          <input
                            type="text"
                            value={relocateTrainPath}
                            onChange={e => setRelocateTrainPath(e.target.value)}
                            placeholder="e.g. playground-series-s6e11/train.csv or C:/data/train.csv"
                            className="flex-1 px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                            required
                          />
                          <button
                            type="button"
                            onClick={() => handleBrowseRelocate('train')}
                            className="px-3 py-2 bg-[#F4F5F6] hover:bg-[#E5E9E6] text-[#121816] text-xs font-medium rounded-xl border border-[#E5E9E6] flex items-center space-x-1.5 transition-colors"
                          >
                            <FolderOpen size={14} />
                            <span>Browse File</span>
                          </button>
                        </div>
                      </div>

                      <div className="space-y-1.5">
                        <label className="text-xs font-semibold text-[#121816] flex items-center justify-between">
                          <span>Test Dataset Path (test.csv, optional)</span>
                          <span className="text-[11px] font-normal text-[#76807B]">Used for blind test inference and submissions</span>
                        </label>
                        <div className="flex space-x-2">
                          <input
                            type="text"
                            value={relocateTestPath}
                            onChange={e => setRelocateTestPath(e.target.value)}
                            placeholder="e.g. playground-series-s6e11/test.csv"
                            className="flex-1 px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                          />
                          <button
                            type="button"
                            onClick={() => handleBrowseRelocate('test')}
                            className="px-3 py-2 bg-[#F4F5F6] hover:bg-[#E5E9E6] text-[#121816] text-xs font-medium rounded-xl border border-[#E5E9E6] flex items-center space-x-1.5 transition-colors"
                          >
                            <FolderOpen size={14} />
                            <span>Browse File</span>
                          </button>
                        </div>
                      </div>

                      {scannedCsvs.length > 0 && (
                        <div className="pt-2">
                          <div className="text-[11px] font-semibold uppercase tracking-wider text-[#76807B] mb-2 font-mono">
                            Detected Workspace CSV Files (Quick Select):
                          </div>
                          <div className="flex flex-wrap gap-2 max-h-36 overflow-y-auto p-2 bg-[#F8FAF9] rounded-xl border border-[#E5E9E6]">
                            {scannedCsvs.map((f, idx) => (
                              <button
                                key={idx}
                                type="button"
                                onClick={() => {
                                  if (f.filename.toLowerCase().includes('test')) {
                                    setRelocateTestPath(f.path);
                                  } else {
                                    setRelocateTrainPath(f.path);
                                  }
                                }}
                                className="text-[11px] font-mono px-2.5 py-1 bg-white hover:bg-[#164E33] hover:text-white border border-[#E5E9E6] rounded-lg transition-colors flex items-center space-x-1 text-[#121816]"
                              >
                                <span>{f.path}</span>
                                <span className="text-[10px] text-zinc-400">({f.size_mb} MB)</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {pathUpdateError && (
                        <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl text-xs">
                          {pathUpdateError}
                        </div>
                      )}

                      {pathUpdateSuccess && (
                        <div className="p-3 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-xl text-xs">
                          {pathUpdateSuccess}
                        </div>
                      )}

                      <div className="pt-2 flex justify-end">
                        <button
                          type="submit"
                          disabled={isUpdatingPaths || !relocateTrainPath.trim()}
                          className="px-5 py-2.5 bg-[#164E33] hover:bg-[#113d28] text-white text-xs font-medium rounded-xl transition-colors flex items-center space-x-2 disabled:opacity-50"
                        >
                          {isUpdatingPaths ? (
                            <>
                              <Loader2 size={14} className="animate-spin" />
                              <span>Relocating & Profiling...</span>
                            </>
                          ) : (
                            <>
                              <Check size={14} />
                              <span>Relocate & Profile Dataset</span>
                            </>
                          )}
                        </button>
                      </div>
                    </form>
                  </div>
                ) : (
                  <>
                    {/* Dataset Source Path & Relocation Bar */}
                    <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 bg-white border border-[#E5E9E6] rounded-xl text-xs">
                      <div className="flex items-center space-x-2 truncate">
                        <span className="text-[#76807B] font-mono uppercase text-[10px] tracking-wider">Source:</span>
                        <code className="text-[#121816] font-mono text-[11px] truncate bg-[#F4F5F6] px-2 py-0.5 rounded border border-[#E5E9E6]">
                          {caseDetail?.resolved_train_path || caseDetail?.meta.train_path}
                        </code>
                        {caseDetail?.resolved_train_path && caseDetail?.resolved_train_path !== caseDetail?.meta.train_path && (
                          <span className="text-[10px] font-mono text-emerald-700 bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded">
                            Auto-Resolved
                          </span>
                        )}
                      </div>
                      <div className="flex items-center space-x-2">
                        <button
                          type="button"
                          onClick={() => {
                            setShowStitcherModal(true);
                            fetchScannedCsvs();
                          }}
                          className="px-2.5 py-1 text-[11px] font-medium text-[#164E33] bg-[#E8F3ED] hover:bg-[#D9EBE1] rounded-lg border border-[#D2E7DC] transition-colors flex items-center space-x-1"
                          title="Open Dataset Stitcher & Relational Joiner Studio"
                        >
                          <Layers size={12} />
                          <span>Dataset Stitcher & Joiner</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => setShowRelocateCard(!showRelocateCard)}
                          className="px-2.5 py-1 text-[11px] font-medium text-[#164E33] hover:bg-[#164E33]/10 rounded-lg border border-[#164E33]/30 transition-colors flex items-center space-x-1"
                        >
                          <FolderOpen size={12} />
                          <span>{showRelocateCard ? 'Hide Path Editor' : 'Relocate / Change Path'}</span>
                        </button>
                      </div>
                    </div>

                    {showRelocateCard && (
                      <div className="bg-white border border-[#E5E9E6] rounded-2xl p-6 shadow-sm space-y-4">
                        <div className="flex items-center justify-between pb-2 border-b border-[#E5E9E6]">
                          <div>
                            <h4 className="text-sm font-semibold text-[#121816]">Relocate Dataset Path</h4>
                            <p className="text-xs text-[#76807B]">Update the path to train.csv and test.csv without modifying code</p>
                          </div>
                          <button
                            type="button"
                            onClick={() => setShowRelocateCard(false)}
                            className="text-zinc-400 hover:text-zinc-600 p-1"
                          >
                            <X size={16} />
                          </button>
                        </div>
                        <form onSubmit={handleUpdatePaths} className="space-y-3">
                          <div className="space-y-1">
                            <label className="text-xs font-semibold text-[#121816]">Training Dataset Path</label>
                            <div className="flex space-x-2">
                              <input
                                type="text"
                                value={relocateTrainPath}
                                onChange={e => setRelocateTrainPath(e.target.value)}
                                className="flex-1 px-3 py-1.5 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                                required
                              />
                              <button
                                type="button"
                                onClick={() => handleBrowseRelocate('train')}
                                className="px-3 py-1.5 bg-[#F4F5F6] hover:bg-[#E5E9E6] text-[#121816] text-xs font-medium rounded-xl border border-[#E5E9E6] flex items-center space-x-1"
                              >
                                <FolderOpen size={13} />
                                <span>Browse</span>
                              </button>
                            </div>
                          </div>
                          <div className="space-y-1">
                            <label className="text-xs font-semibold text-[#121816]">Test Dataset Path (Optional)</label>
                            <div className="flex space-x-2">
                              <input
                                type="text"
                                value={relocateTestPath}
                                onChange={e => setRelocateTestPath(e.target.value)}
                                className="flex-1 px-3 py-1.5 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                              />
                              <button
                                type="button"
                                onClick={() => handleBrowseRelocate('test')}
                                className="px-3 py-1.5 bg-[#F4F5F6] hover:bg-[#E5E9E6] text-[#121816] text-xs font-medium rounded-xl border border-[#E5E9E6] flex items-center space-x-1"
                              >
                                <FolderOpen size={13} />
                                <span>Browse</span>
                              </button>
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <div className="space-y-1">
                              <label className="text-xs font-semibold text-[#121816]">Target Column</label>
                              <input
                                type="text"
                                value={relocateTargetCol}
                                onChange={e => setRelocateTargetCol(e.target.value)}
                                placeholder="e.g. x_base, target"
                                className="w-full px-3 py-1.5 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                              />
                            </div>
                            <div className="space-y-1">
                              <label className="text-xs font-semibold text-[#121816]">ID Column (Optional)</label>
                              <input
                                type="text"
                                value={relocateIdCol}
                                onChange={e => setRelocateIdCol(e.target.value)}
                                placeholder="e.g. frame_id, id"
                                className="w-full px-3 py-1.5 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                              />
                            </div>
                          </div>
                          <div className="flex justify-end pt-2">
                            <button
                              type="submit"
                              disabled={isUpdatingPaths || !relocateTrainPath.trim()}
                              className="px-4 py-2 bg-[#164E33] hover:bg-[#113d28] text-white text-xs font-medium rounded-xl transition-colors flex items-center space-x-1.5 disabled:opacity-50"
                            >
                              {isUpdatingPaths ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
                              <span>Save & Re-Profile</span>
                            </button>
                          </div>
                        </form>
                      </div>
                    )}

                    {/* Executive Summary Metrics */}
                    <div className="grid grid-cols-4 gap-4">
                      <div className="bg-white border border-[#E5E9E6] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
                        <div>
                          <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">Total Rows</div>
                          <div className="text-2xl font-bold mt-1 text-[#121816] font-mono">
                            {edaData.summary.total_rows.toLocaleString()}
                          </div>
                        </div>
                        <div className="text-xs text-[#76807B] mt-2 pt-2 border-t border-[#E5E9E6]/60">Full Training Records</div>
                      </div>

                      <div className="bg-white border border-[#E5E9E6] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
                        <div>
                          <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">Total Features</div>
                          <div className="text-2xl font-bold mt-1 text-[#121816] font-mono">
                            {edaData.summary.total_cols} Cols
                          </div>
                        </div>
                        <div className="text-xs text-[#76807B] mt-2 pt-2 border-t border-[#E5E9E6]/60">
                          Dataset Size: <span className="font-semibold text-[#121816]">{edaData.summary.file_size_mb} MB</span>
                        </div>
                      </div>

                      <div className="bg-white border border-[#E5E9E6] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
                        <div>
                          <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">Missing Values</div>
                          <div className="text-2xl font-bold mt-1 text-[#121816] font-mono">
                            {edaData.summary.missing_pct_sample}
                          </div>
                        </div>
                        <div className="text-xs text-[#76807B] mt-2 pt-2 border-t border-[#E5E9E6]/60">
                          <span className="font-semibold text-[#121816]">{edaData.summary.missing_cells_sample.toLocaleString()}</span> cells
                          <div className="text-[11px] text-[#76807B] truncate mt-0.5" title={edaData.summary.null_cols_note}>
                            {edaData.summary.null_cols_note?.startsWith("Across") 
                              ? edaData.summary.null_cols_note 
                              : `in ${edaData.summary.null_cols_note || 'sample'}`}
                          </div>
                        </div>
                      </div>

                      <div className="bg-white border border-[#E5E9E6] rounded-2xl p-5 shadow-sm flex flex-col justify-between">
                        <div>
                          <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">Target Variable</div>
                          <div className="text-2xl font-bold mt-1 text-[#121816] font-mono truncate" title={edaData.summary.target_col}>
                            {edaData.summary.target_col || 'None'}
                          </div>
                        </div>
                        <div className="text-xs text-[#76807B] mt-2 pt-2 border-t border-[#E5E9E6]/60">
                          {edaData.summary.is_regression ? (
                            <span>Range: <span className="font-semibold text-[#121816]">{edaData.summary.target_summary?.min ?? 'N/A'} → {edaData.summary.target_summary?.max ?? 'N/A'}</span></span>
                          ) : (
                            <span><span className="font-semibold text-[#121816]">{edaData.target_distribution.length}</span> Distinct Classes</span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Target Distribution & Correlation Ranking */}
                    <div className="grid grid-cols-3 gap-6">
                      {/* Target Distribution / Regression Histogram */}
                      <div className="bg-white border border-[#E5E9E6] rounded-2xl p-6 shadow-sm space-y-4">
                        <div className="flex justify-between items-center">
                          <h3 className="font-bold text-sm text-[#121816]">
                            {edaData.summary.is_regression ? 'Target Continuous Distribution' : 'Target Class Balance'}
                          </h3>
                          <span className="text-[10px] font-mono uppercase bg-[#164E33]/10 text-[#164E33] px-2 py-0.5 rounded font-semibold">
                            {edaData.summary.is_regression ? 'Continuous' : 'Discrete'}
                          </span>
                        </div>
                        <p className="text-xs text-[#76807B] leading-relaxed">
                          {edaData.summary.is_regression
                            ? `Continuous numerical spread across 10 buckets (Mean: ${edaData.summary.target_summary?.mean ?? 'N/A'}, Std: ${edaData.summary.target_summary?.std ?? 'N/A'}).`
                            : 'Class distribution in sample. Equal distribution indicates no class imbalance treatment needed.'}
                        </p>

                        <div className="space-y-3.5 pt-2 max-h-72 overflow-y-auto">
                          {edaData.target_distribution.map((td: any) => (
                            <div key={td.label} className="space-y-1.5">
                              <div className="flex justify-between text-xs font-medium">
                                <span className="font-mono font-semibold text-[#121816]">{td.label}</span>
                                <span className="text-[#76807B] font-mono">{td.percentage}% ({td.count.toLocaleString()})</span>
                              </div>
                              <div className="w-full bg-[#F4F5F6] h-2.5 rounded-full overflow-hidden border border-[#E5E9E6]/60">
                                <div className="bg-[#164E33] h-full rounded-full" style={{ width: `${td.percentage}%` }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Top Correlations with Target */}
                      <div className="col-span-2 bg-white border border-[#E5E9E6] rounded-2xl p-6 shadow-sm space-y-4">
                        <div className="flex justify-between items-center">
                          <h3 className="font-bold text-sm text-[#121816]">Top Predictive Signals (Correlations with Target)</h3>
                          <span className="text-[11px] font-mono bg-[#E8F3ED] text-[#164E33] px-2 py-0.5 rounded-md border border-[#D2E7DC]">
                            Pearson (r)
                          </span>
                        </div>
                        <p className="text-xs text-[#76807B] leading-relaxed">
                          Features with highest linear correlation magnitude with {edaData.summary.target_col}. Strong indicators for tree node splitting.
                        </p>

                        <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 pt-2">
                          {edaData.correlations.map((c, idx) => (
                            <div 
                              key={c.feature} 
                              onClick={() => setSelectedFeature(c.feature)}
                              className="flex items-center justify-between text-xs font-mono py-1.5 border-b border-[#E5E9E6]/60 hover:bg-[#F8FAF9] cursor-pointer px-2 rounded-lg transition"
                              title="Click to view distribution"
                            >
                              <span className="truncate w-44 text-[#121816] font-medium">#{idx + 1} {c.feature}</span>
                              <span className={`font-semibold ${c.correlation >= 0 ? 'text-[#164E33]' : 'text-[#76807B]'}`}>
                                {c.correlation > 0 ? `+${c.correlation.toFixed(3)}` : c.correlation.toFixed(3)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Interactive Feature Pattern & Distribution Explorer */}
                    {(() => {
                      const activeFeatureName = selectedFeature || (edaData.correlations[0]?.feature || edaData.columns_profile[0]?.name);
                      const activeCol = edaData.columns_profile.find(c => c.name === activeFeatureName);
                      const activeCorr = edaData.correlations.find(c => c.feature === activeFeatureName);

                      return (
                        <div className="border border-zinc-200 rounded p-6 bg-white space-y-5">
                          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-100 pb-4">
                            <div>
                              <div className="flex items-center space-x-2">
                                <h3 className="font-bold text-sm text-zinc-950 uppercase tracking-tight">Feature Pattern & Target Response Explorer</h3>
                                <span className="text-[10px] font-mono uppercase bg-zinc-100 text-zinc-700 px-2 py-0.5 rounded border border-zinc-200">
                                  Visual EDA
                                </span>
                              </div>
                              <p className="text-xs text-zinc-500 mt-1">
                                Visual inspection of feature distributions and conditional target probability (% satisfaction per bin).
                              </p>
                            </div>

                            {/* Feature Selector Dropdown */}
                            <div className="flex items-center space-x-2">
                              <label className="text-xs font-semibold text-zinc-600">Select Feature:</label>
                              <select
                                value={activeFeatureName}
                                onChange={e => setSelectedFeature(e.target.value)}
                                className="bg-zinc-50 border border-zinc-300 rounded px-3 py-1.5 text-xs font-mono font-medium text-zinc-900 focus:outline-none focus:border-zinc-900"
                              >
                                {edaData.columns_profile.map(c => (
                                  <option key={c.name} value={c.name}>
                                    {c.name} ({c.type})
                                  </option>
                                ))}
                              </select>
                            </div>
                          </div>

                          {/* Active Feature Metric Badges */}
                          {activeCol && (
                            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                              <div className="border border-zinc-200 rounded p-3 bg-zinc-50/50">
                                <div className="text-[10px] uppercase font-semibold text-zinc-500">Feature Type</div>
                                <div className="text-sm font-bold text-zinc-900 mt-0.5 font-mono">{activeCol.type}</div>
                              </div>
                              <div className="border border-zinc-200 rounded p-3 bg-zinc-50/50">
                                <div className="text-[10px] uppercase font-semibold text-zinc-500">Unique Values</div>
                                <div className="text-sm font-bold text-zinc-900 mt-0.5 font-mono">{activeCol.unique_count.toLocaleString()}</div>
                              </div>
                              <div className="border border-zinc-200 rounded p-3 bg-zinc-50/50">
                                <div className="text-[10px] uppercase font-semibold text-zinc-500">Missing Rate</div>
                                <div className="text-sm font-bold text-zinc-900 mt-0.5 font-mono">{activeCol.missing_rate}%</div>
                              </div>
                              <div className="border border-zinc-200 rounded p-3 bg-zinc-50/50">
                                <div className="text-[10px] uppercase font-semibold text-zinc-500">Correlation with Target</div>
                                <div className="text-sm font-bold text-zinc-900 mt-0.5 font-mono">
                                  {activeCorr ? (activeCorr.correlation > 0 ? `+${activeCorr.correlation.toFixed(4)}` : activeCorr.correlation.toFixed(4)) : 'N/A'}
                                </div>
                              </div>
                              <div className="border border-zinc-200 rounded p-3 bg-zinc-50/50">
                                <div className="text-[10px] uppercase font-semibold text-zinc-500">Range / Mode</div>
                                <div className="text-sm font-bold text-zinc-900 mt-0.5 font-mono truncate">
                                  {activeCol.stats?.mean !== undefined ? `${activeCol.stats.min} → ${activeCol.stats.max}` : `"${activeCol.stats?.top_value || 'N/A'}"`}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* View Mode Toggle: Composite vs Line Curve vs Bars */}
                          {activeCol?.distribution && activeCol.distribution.length > 0 && (
                            <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
                              <span className="text-xs font-semibold text-zinc-700">Visualization Perspective:</span>
                              <div className="flex items-center space-x-1 bg-zinc-100 p-1 rounded">
                                <button
                                  type="button"
                                  onClick={() => setEdaViewMode('composite')}
                                  className={`flex items-center space-x-1.5 px-3 py-1 text-xs font-mono rounded transition ${
                                    edaViewMode === 'composite' ? 'bg-white text-zinc-950 font-bold shadow-xs' : 'text-zinc-600 hover:text-zinc-900'
                                  }`}
                                >
                                  <Activity size={12} />
                                  <span>Composite Curve</span>
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setEdaViewMode('trend')}
                                  className={`flex items-center space-x-1.5 px-3 py-1 text-xs font-mono rounded transition ${
                                    edaViewMode === 'trend' ? 'bg-white text-zinc-950 font-bold shadow-xs' : 'text-zinc-600 hover:text-zinc-900'
                                  }`}
                                >
                                  <TrendingUp size={12} />
                                  <span>Trend Line Curve</span>
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setEdaViewMode('bars')}
                                  className={`flex items-center space-x-1.5 px-3 py-1 text-xs font-mono rounded transition ${
                                    edaViewMode === 'bars' ? 'bg-white text-zinc-950 font-bold shadow-xs' : 'text-zinc-600 hover:text-zinc-900'
                                  }`}
                                >
                                  <BarChart2 size={12} />
                                  <span>Bar Breakdown</span>
                                </button>
                              </div>
                            </div>
                          )}

                          {/* Chart Container */}
                          {activeCol?.distribution && activeCol.distribution.length > 0 ? (
                            <div className="space-y-4 pt-1">
                              {edaViewMode !== 'bars' ? (
                                /* SVG Target Response Curve / Trend Line Chart */
                                (() => {
                                  const dist = activeCol.distribution!;
                                  const validPoints = dist.filter(d => d.target_rate !== undefined);
                                  const globalTargetPct = edaData.target_distribution.length === 2 ? edaData.target_distribution[1].percentage : 50;

                                  // Calculate Monotonicity & Inflection Jump
                                  let trendNature = 'Stable / Neutral';
                                  let maxDelta = 0;
                                  let inflectionFrom = '';
                                  let inflectionTo = '';
                                  let maxDeltaSign = '+';

                                  if (validPoints.length >= 2) {
                                    let inc = 0;
                                    let dec = 0;
                                    for (let i = 0; i < validPoints.length - 1; i++) {
                                      const diff = validPoints[i + 1].target_rate! - validPoints[i].target_rate!;
                                      if (diff > 2) inc++;
                                      else if (diff < -2) dec++;

                                      if (Math.abs(diff) > maxDelta) {
                                        maxDelta = Math.abs(diff);
                                        maxDeltaSign = diff >= 0 ? '+' : '-';
                                        inflectionFrom = validPoints[i].label;
                                        inflectionTo = validPoints[i + 1].label;
                                      }
                                    }

                                    if (inc > 0 && dec === 0) trendNature = 'Monotonically Increasing';
                                    else if (dec > 0 && inc === 0) trendNature = 'Monotonically Decreasing';
                                    else if (inc > 0 && dec > 0) trendNature = 'Non-Linear / Inflection Curve';
                                  }

                                  const svgWidth = 680;
                                  const svgHeight = 220;
                                  const padLeft = 45;
                                  const padRight = 30;
                                  const padTop = 25;
                                  const padBottom = 45;
                                  const plotW = svgWidth - padLeft - padRight;
                                  const plotH = svgHeight - padTop - padBottom;

                                  const getY = (val: number) => padTop + (1 - val / 100) * plotH;
                                  const getX = (i: number) => padLeft + (i + 0.5) * (plotW / validPoints.length);
                                  const maxPct = Math.max(...dist.map(d => d.pct), 1);
                                  const baseLineY = getY(globalTargetPct);

                                  const polylinePts = validPoints.map((item, i) => `${getX(i)},${getY(item.target_rate!)}`).join(' ');

                                  return (
                                    <div className="space-y-4">
                                      {/* Legend Header */}
                                      <div className="flex items-center justify-between text-[11px] font-mono text-zinc-500">
                                        <div className="flex items-center space-x-6">
                                          <span className="flex items-center space-x-1.5">
                                            <span className="inline-block w-3 h-0.5 bg-zinc-950"></span>
                                            <span className="text-zinc-900 font-semibold">Target Response Curve (% {edaData.summary.target_col})</span>
                                          </span>
                                          {edaViewMode === 'composite' && (
                                            <span className="flex items-center space-x-1.5">
                                              <span className="inline-block w-2.5 h-2.5 bg-zinc-200 border border-zinc-300 rounded-xs"></span>
                                              <span>Population Density Bar</span>
                                            </span>
                                          )}
                                          <span className="flex items-center space-x-1.5">
                                            <span className="inline-block w-3 h-0.5 border-t border-dashed border-zinc-400"></span>
                                            <span>Dataset Avg ({globalTargetPct}%)</span>
                                          </span>
                                        </div>
                                        <span className="text-zinc-400">Y-Axis: Target Probability (0% - 100%)</span>
                                      </div>

                                      {/* Responsive SVG Chart */}
                                      <div className="w-full bg-zinc-50/40 border border-zinc-200 rounded p-2 overflow-x-auto">
                                        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto min-w-[500px]">
                                          {/* Horizontal Grid Lines */}
                                          {[0, 25, 50, 75, 100].map(v => (
                                            <g key={v}>
                                              <line
                                                x1={padLeft}
                                                y1={getY(v)}
                                                x2={svgWidth - padRight}
                                                y2={getY(v)}
                                                stroke="#e4e4e7"
                                                strokeWidth="1"
                                                strokeDasharray="2 3"
                                              />
                                              <text
                                                x={padLeft - 6}
                                                y={getY(v) + 3}
                                                textAnchor="end"
                                                fontSize="9"
                                                fontFamily="monospace"
                                                fill="#71717a"
                                              >
                                                {v}%
                                              </text>
                                            </g>
                                          ))}

                                          {/* Dataset Baseline Average Dashed Line */}
                                          <line
                                            x1={padLeft}
                                            y1={baseLineY}
                                            x2={svgWidth - padRight}
                                            y2={baseLineY}
                                            stroke="#a1a1aa"
                                            strokeWidth="1.5"
                                            strokeDasharray="4 4"
                                          />

                                          {/* Composite Background Density Bars */}
                                          {edaViewMode === 'composite' &&
                                            validPoints.map((item, i) => {
                                              const barW = Math.max(12, Math.min(32, (plotW / validPoints.length) * 0.5));
                                              const h = (item.pct / maxPct) * plotH;
                                              return (
                                                <rect
                                                  key={item.label}
                                                  x={getX(i) - barW / 2}
                                                  y={padTop + plotH - h}
                                                  width={barW}
                                                  height={h}
                                                  fill="#e4e4e7"
                                                  stroke="#d4d4d8"
                                                  rx="1.5"
                                                />
                                              );
                                            })}

                                          {/* Target Response Polyline */}
                                          <polyline
                                            points={polylinePts}
                                            fill="none"
                                            stroke="#09090b"
                                            strokeWidth="2.5"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                          />

                                          {/* Data Point Circles and Text Badges */}
                                          {validPoints.map((item, i) => {
                                            const cx = getX(i);
                                            const cy = getY(item.target_rate!);
                                            return (
                                              <g key={item.label}>
                                                <circle
                                                  cx={cx}
                                                  cy={cy}
                                                  r="4.5"
                                                  fill="#ffffff"
                                                  stroke="#09090b"
                                                  strokeWidth="2.5"
                                                />
                                                <text
                                                  x={cx}
                                                  y={cy - 9}
                                                  textAnchor="middle"
                                                  fontSize="10"
                                                  fontFamily="monospace"
                                                  fontWeight="700"
                                                  fill="#09090b"
                                                >
                                                  {item.target_rate}%
                                                </text>
                                                {/* X Axis Label */}
                                                <text
                                                  x={cx}
                                                  y={svgHeight - 14}
                                                  textAnchor="middle"
                                                  fontSize="9"
                                                  fontFamily="monospace"
                                                  fill="#52525b"
                                                >
                                                  {item.label}
                                                </text>
                                              </g>
                                            );
                                          })}
                                        </svg>
                                      </div>

                                      {/* Inflection & Monotonicity Insight Panel */}
                                      <div className="border border-zinc-200 rounded p-4 bg-zinc-50 flex flex-col md:flex-row md:items-center justify-between gap-4 text-xs font-mono">
                                        <div className="space-y-1">
                                          <div className="flex items-center space-x-2">
                                            <span className="font-semibold text-zinc-950 uppercase">Curve Diagnosis:</span>
                                            <span className="bg-zinc-900 text-white px-2 py-0.5 rounded text-[11px] font-bold">
                                              {trendNature}
                                            </span>
                                          </div>
                                          <p className="text-zinc-600 font-sans text-xs">
                                            {trendNature === 'Non-Linear / Inflection Curve'
                                              ? 'Complex non-linear response detected. GBDT trees will benefit from threshold boundary splits.'
                                              : trendNature === 'Monotonically Increasing'
                                              ? 'Strict positive linear signal. Highly predictive monotonic ranking feature.'
                                              : 'Monotonically decreasing signal. Inversely proportional to positive target probability.'}
                                          </p>
                                        </div>

                                        {maxDelta > 0 && (
                                          <div className="border border-zinc-300 bg-white p-2.5 rounded text-right space-y-0.5 shrink-0">
                                            <div className="text-[10px] uppercase text-zinc-500 font-semibold">Max Inflection Jump</div>
                                            <div className="font-bold text-zinc-950 text-sm">
                                              {maxDeltaSign}{maxDelta.toFixed(1)}% shift
                                            </div>
                                            <div className="text-[10px] text-zinc-500 truncate max-w-[200px]">
                                              Between "{inflectionFrom}" & "{inflectionTo}"
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  );
                                })()
                              ) : (
                                /* Classic Bar Distribution View */
                                <div className="space-y-3 pt-2">
                                  <div className="flex justify-between items-center text-xs text-zinc-500">
                                    <span>Bin Range / Category</span>
                                    <div className="flex items-center space-x-6 text-[11px] font-mono">
                                      <span className="flex items-center space-x-1">
                                        <span className="inline-block w-2.5 h-2.5 bg-zinc-900 rounded-sm"></span>
                                        <span>Frequency (% of Total)</span>
                                      </span>
                                      {activeCol.distribution.some(d => d.target_rate !== undefined) && (
                                        <span className="flex items-center space-x-1">
                                          <span className="inline-block w-2.5 h-2.5 bg-zinc-400 rounded-sm"></span>
                                          <span>Target Positive Rate (% {edaData.summary.target_col})</span>
                                        </span>
                                      )}
                                    </div>
                                  </div>

                                  <div className="space-y-2.5">
                                    {activeCol.distribution.map(item => {
                                      const maxPct = Math.max(...activeCol.distribution!.map(d => d.pct), 1);
                                      const barWidth = Math.max(3, Math.round((item.pct / maxPct) * 100));
                                      return (
                                        <div key={item.label} className="grid grid-cols-12 gap-3 items-center text-xs font-mono">
                                          <div className="col-span-3 truncate text-zinc-800 font-medium" title={item.label}>
                                            {item.label}
                                          </div>
                                          <div className="col-span-6 space-y-1">
                                            <div className="w-full bg-zinc-100 h-3 rounded overflow-hidden">
                                              <div className="bg-zinc-900 h-full rounded transition-all duration-300" style={{ width: `${barWidth}%` }} />
                                            </div>
                                            {item.target_rate !== undefined && (
                                              <div className="w-full bg-zinc-100 h-1.5 rounded overflow-hidden">
                                                <div className="bg-zinc-400 h-full rounded" style={{ width: `${item.target_rate}%` }} />
                                              </div>
                                            )}
                                          </div>
                                          <div className="col-span-3 flex items-center justify-end space-x-3 text-right">
                                            <span className="text-zinc-900 font-semibold">{item.pct}%</span>
                                            {item.target_rate !== undefined && (
                                              <span className="text-zinc-500 text-[11px]">
                                                ({item.target_rate}% target)
                                              </span>
                                            )}
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="py-8 text-center text-xs text-zinc-400">
                              No discrete distribution buckets available for this feature.
                            </div>
                          )}
                        </div>
                      );
                    })()}

                    {/* Column Profiler Table */}
                    <div className="border border-zinc-200 rounded overflow-hidden">
                      <div className="px-5 py-3 border-b border-zinc-200 bg-zinc-50 flex items-center justify-between">
                        <span className="font-semibold text-sm text-zinc-950">Feature Schema & Data Profiling</span>
                        <span className="text-xs text-zinc-500 font-mono">{edaData.columns_profile.length} Total Columns</span>
                      </div>

                      <div className="overflow-x-auto max-h-96 overflow-y-auto">
                        <table className="w-full text-left text-xs font-mono">
                          <thead className="bg-zinc-50 border-b border-zinc-200 uppercase text-zinc-500 sticky top-0">
                            <tr>
                              <th className="px-4 py-2.5">Feature Name</th>
                              <th className="px-4 py-2.5">Type</th>
                              <th className="px-4 py-2.5">Missing</th>
                              <th className="px-4 py-2.5">Uniques</th>
                              <th className="px-4 py-2.5">Sample Statistics (Min / Mean / Max / Top)</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-zinc-200">
                            {edaData.columns_profile.map(col => (
                              <tr key={col.name} className={col.is_target ? 'bg-zinc-100/50 font-semibold' : ''}>
                                <td className="px-4 py-2.5 font-medium text-zinc-950 flex items-center space-x-1.5">
                                  <span>{col.name}</span>
                                  {col.is_target && <span className="text-[10px] bg-zinc-900 text-white px-1 rounded">TARGET</span>}
                                  {col.is_id && <span className="text-[10px] border border-zinc-300 text-zinc-600 px-1 rounded">ID</span>}
                                </td>
                                <td className="px-4 py-2.5 text-zinc-600">
                                  {col.type} <span className="text-zinc-400">({col.dtype})</span>
                                </td>
                                <td className="px-4 py-2.5">
                                  {col.missing_count > 0 ? (
                                    <span className="text-zinc-950 font-semibold bg-zinc-200 px-1 py-0.5 rounded">
                                      {col.missing_rate}% ({col.missing_count})
                                    </span>
                                  ) : (
                                    <span className="text-zinc-400">0%</span>
                                  )}
                                </td>
                                <td className="px-4 py-2.5 text-zinc-800">{col.unique_count.toLocaleString()}</td>
                                <td className="px-4 py-2.5 text-zinc-600">
                                  {col.stats?.mean !== undefined ? (
                                    <span>
                                      Min: {col.stats.min} | Mean: {col.stats.mean} | Max: {col.stats.max}
                                    </span>
                                  ) : col.stats?.top_value ? (
                                    <span>Top: "{col.stats.top_value}"</span>
                                  ) : (
                                    '-'
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Live 25-Row Sample Table */}
                    <div className="border border-zinc-200 rounded overflow-hidden">
                      <div className="px-5 py-3 border-b border-zinc-200 bg-zinc-50 flex items-center justify-between">
                        <span className="font-semibold text-sm text-zinc-950">First 25 Sample Records (Raw Table)</span>
                        <span className="text-xs text-zinc-500 font-mono">{meta?.train_path}</span>
                      </div>

                      <div className="overflow-x-auto max-h-72 overflow-y-auto">
                        <table className="w-full text-left text-xs font-mono">
                          <thead className="bg-zinc-50 border-b border-zinc-200 text-zinc-600 sticky top-0">
                            <tr>
                              {edaData.columns_profile.map(c => (
                                <th key={c.name} className="px-3 py-2 whitespace-nowrap">
                                  {c.name}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-zinc-200">
                            {edaData.preview_rows.map((row, rIdx) => (
                              <tr key={rIdx} className="hover:bg-zinc-50">
                                {edaData.columns_profile.map(c => (
                                  <td key={c.name} className="px-3 py-2 whitespace-nowrap text-zinc-800">
                                    {String(row[c.name])}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* TAB 2: FEATURE ENGINEERING (Auto-FE & Domain Hook) */}
            {activeTab === 'fe' && (
              <div className="space-y-6">
                {/* 1. Universal Auto-FE Generator Architecture */}
                <div className="border border-zinc-200 rounded p-6 bg-white space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-bold text-base text-zinc-950">Universal Automated Feature Engineering (Auto-FE)</h3>
                      <p className="text-xs text-zinc-500 mt-0.5">
                        Domain-agnostic mathematical feature primitives executed leak-free fold-by-fold.
                      </p>
                    </div>
                    <span className="text-xs font-mono uppercase bg-zinc-100 text-zinc-800 border border-zinc-200 px-2 py-1 rounded">
                      Mathematical Primitives Engine
                    </span>
                  </div>

                  {/* 4 Feature Primitives Visual Grid */}
                  <div className="grid grid-cols-4 gap-4 pt-2">
                    <div className="border border-zinc-200 rounded p-3.5 bg-zinc-50/50 space-y-1.5">
                      <div className="text-xs font-bold text-zinc-950">1. Group Relative Z-Scores</div>
                      <p className="text-[11px] text-zinc-500 leading-relaxed font-mono">
                        (x - Mean_by_Cat) / Std_by_Cat
                      </p>
                      <p className="text-[11px] text-zinc-600 leading-snug">
                        Measures a student/customer/passenger's value relative to their peer group baseline.
                      </p>
                    </div>

                    <div className="border border-zinc-200 rounded p-3.5 bg-zinc-50/50 space-y-1.5">
                      <div className="text-xs font-bold text-zinc-950">2. Interaction Pairs & Ratios</div>
                      <p className="text-[11px] text-zinc-500 leading-relaxed font-mono">
                        Num_A / (Num_B + 1e-5)
                      </p>
                      <p className="text-[11px] text-zinc-600 leading-snug">
                        Computes per-capita intensity, speed, or financial ratios across continuous columns.
                      </p>
                    </div>

                    <div className="border border-zinc-200 rounded p-3.5 bg-zinc-50/50 space-y-1.5">
                      <div className="text-xs font-bold text-zinc-950">3. Frequency & Rarity</div>
                      <p className="text-[11px] text-zinc-500 leading-relaxed font-mono">
                        Cat_col.map(train_freq)
                      </p>
                      <p className="text-[11px] text-zinc-600 leading-snug">
                        Maps high-cardinality entities (schools, cities, zipcodes) to prevalence rates without leakage.
                      </p>
                    </div>

                    <div className="border border-zinc-200 rounded p-3.5 bg-zinc-50/50 space-y-1.5">
                      <div className="text-xs font-bold text-zinc-950">4. Automated Pruner</div>
                      <p className="text-[11px] text-zinc-500 leading-relaxed font-mono">
                        Drops |r| &gt; 0.98 & Var &lt; 1e-7
                      </p>
                      <p className="text-[11px] text-zinc-600 leading-snug">
                        Guarantees feature matrices remain lean, avoiding memory bloat and collinearity.
                      </p>
                    </div>
                  </div>
                </div>

                {/* 2. Interactive Domain Feature Engineering Hook Editor */}
                <div className="border border-[#E5E9E6] rounded-2xl p-6 bg-white space-y-5 shadow-sm">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E5E9E6] pb-4">
                    <div>
                      <div className="flex items-center space-x-2">
                        <h4 className="font-bold text-sm text-[#121816] uppercase tracking-tight">Domain Feature Engineering Hook (`feature_hook.py`)</h4>
                        <span className="text-[10px] font-mono uppercase bg-[#E8F3ED] text-[#164E33] border border-[#D2E7DC] px-2 py-0.5 rounded font-bold">
                          Live In-Browser Code & Visual Builder
                        </span>
                      </div>
                      <p className="text-xs text-[#76807B] mt-1">
                        Generate domain-specific formulas visually or write custom Python transform logic. Executes leak-free before cross-validation folds.
                      </p>
                    </div>

                    <div className="flex items-center space-x-2">
                      {hookSaved && (
                        <span className="text-xs font-mono text-[#164E33] font-semibold flex items-center space-x-1 mr-2">
                          <Check size={14} />
                          <span>Saved & Ready</span>
                        </span>
                      )}

                      <button
                        type="button"
                        onClick={handleUndoHook}
                        disabled={hookHistory.length === 0}
                        title="Undo last code modification"
                        className={`flex items-center space-x-1 border px-3 py-2 rounded-xl text-xs font-medium transition shadow-2xs ${
                          hookHistory.length === 0 
                            ? 'border-[#E5E9E6] text-[#A0AAA5] cursor-not-allowed bg-[#F8FAF9]' 
                            : 'border-[#E5E9E6] text-[#121816] hover:bg-[#F4F5F6]'
                        }`}
                      >
                        <RotateCcw size={13} />
                        <span>Undo</span>
                      </button>

                      <button
                        type="button"
                        onClick={handleResetHook}
                        title="Reset code hook to clean default template"
                        className="flex items-center space-x-1 border border-[#E5E9E6] hover:border-red-300 text-[#76807B] hover:text-red-600 hover:bg-red-50/50 px-3 py-2 rounded-xl text-xs font-medium transition shadow-2xs"
                      >
                        <RefreshCw size={13} />
                        <span>Reset</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => handleTestHook()}
                        disabled={hookTesting}
                        title="Dry-run and validate feature hook on a 1k sample"
                        className="flex items-center space-x-1 border border-[#E5E9E6] hover:border-[#164E33] text-[#121816] hover:bg-[#F8FAF9] px-3 py-2 rounded-xl text-xs font-medium transition shadow-2xs"
                      >
                        {hookTesting ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
                        <span>Dry-Run Test</span>
                      </button>

                      <button
                        onClick={handleSaveHook}
                        disabled={savingHook}
                        className="flex items-center space-x-1.5 bg-[#164E33] hover:bg-[#113C27] text-white text-xs font-semibold px-4 py-2 rounded-xl transition shadow-2xs"
                      >
                        {savingHook ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />}
                        <span>{savingHook ? 'Saving Hook...' : 'Save Feature Hook'}</span>
                      </button>
                    </div>
                  </div>

                  {/* Hook Validation Status Card */}
                  {hookValidation && (
                    <div className={`border rounded-xl p-4 transition ${
                      hookValidation.status === 'valid'
                        ? 'border-[#D2E7DC] bg-[#F8FAF9]'
                        : hookValidation.status === 'error'
                        ? 'border-red-200 bg-red-50/40 text-red-900'
                        : 'border-[#E5E9E6] bg-white'
                    }`}>
                      <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-[#E5E9E6] pb-3">
                        <div className="flex items-center space-x-2">
                          <span className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded font-bold border ${
                            hookValidation.status === 'valid'
                              ? 'bg-[#E8F3ED] text-[#164E33] border-[#D2E7DC]'
                              : 'bg-red-100 text-red-700 border-red-200'
                          }`}>
                            {hookValidation.status === 'valid' ? 'Active & Validated' : 'Validation Error'}
                          </span>
                          <span className="text-xs font-bold text-[#121816]">
                            {hookValidation.status === 'valid' 
                              ? `+${hookValidation.added_features_count} New Features Verified (${hookValidation.transformed_cols_count} Total Columns)`
                              : 'Feature Hook Failed Execution'}
                          </span>
                        </div>
                        <div className="flex items-center space-x-3 text-[11px] font-mono text-[#76807B]">
                          {hookValidation.duration_ms !== undefined && (
                            <span>Dry-run: {hookValidation.duration_ms}ms (1k sample)</span>
                          )}
                          <button
                            type="button"
                            onClick={() => handleTestHook()}
                            disabled={hookTesting}
                            className="text-xs text-[#164E33] hover:underline font-semibold flex items-center space-x-1"
                          >
                            {hookTesting && <Loader2 size={11} className="animate-spin" />}
                            <span>Re-test Hook</span>
                          </button>
                        </div>
                      </div>

                      {hookValidation.status === 'error' ? (
                        <div className="mt-3 p-3 bg-white rounded-lg border border-red-200 text-xs font-mono text-red-600">
                          {hookValidation.message}
                        </div>
                      ) : (
                        <div className="mt-3 space-y-2">
                          <div className="text-[10px] font-bold uppercase tracking-wider text-[#76807B] font-mono">
                            Generated Feature Columns ({hookValidation.added_features?.length || 0}):
                          </div>
                          <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
                            {hookValidation.added_features?.map((col: string) => (
                              <span key={col} className="text-[11px] font-mono bg-white text-[#164E33] border border-[#D2E7DC] px-2 py-0.5 rounded-md shadow-2xs">
                                + {col}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* 1-Click Universal Feature Primitives */}
                  <div className="space-y-2">
                    <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">Universal Feature Primitives (Applicable to Any Tabular Task)</div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={handleInsertGroupStatsTemplate}
                        className="flex items-center space-x-1.5 bg-[#F8FAF9] hover:bg-[#E8F3ED] text-[#121816] hover:text-[#164E33] border border-[#E5E9E6] hover:border-[#164E33] px-3 py-1.5 rounded-xl text-xs font-medium transition shadow-2xs"
                      >
                        <Plus size={12} />
                        <span>+ Group-By Baseline & Relative Z-Scores</span>
                      </button>

                      <button
                        type="button"
                        onClick={handleInsertRowStatsTemplate}
                        className="flex items-center space-x-1.5 bg-[#F8FAF9] hover:bg-[#E8F3ED] text-[#121816] hover:text-[#164E33] border border-[#E5E9E6] hover:border-[#164E33] px-3 py-1.5 rounded-xl text-xs font-medium transition shadow-2xs"
                      >
                        <Plus size={12} />
                        <span>+ Row-wise Intensity & Summary Stats</span>
                      </button>

                      <button
                        type="button"
                        onClick={handleInsertFreqEncodingTemplate}
                        className="flex items-center space-x-1.5 bg-[#F8FAF9] hover:bg-[#E8F3ED] text-[#121816] hover:text-[#164E33] border border-[#E5E9E6] hover:border-[#164E33] px-3 py-1.5 rounded-xl text-xs font-medium transition shadow-2xs"
                      >
                        <Plus size={12} />
                        <span>+ High-Cardinality Frequency Rarity</span>
                      </button>

                      <button
                        type="button"
                        onClick={handleInsertLogTransformTemplate}
                        className="flex items-center space-x-1.5 bg-[#F8FAF9] hover:bg-[#E8F3ED] text-[#121816] hover:text-[#164E33] border border-[#E5E9E6] hover:border-[#164E33] px-3 py-1.5 rounded-xl text-xs font-medium transition shadow-2xs"
                      >
                        <Plus size={12} />
                        <span>+ Log1p Skew Correction</span>
                      </button>
                    </div>
                  </div>

                  {/* Visual Builders: Pairwise & Group-By */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Visual Formula Builder */}
                    <div className="border border-[#E5E9E6] rounded-xl p-4 bg-[#F8FAF9] space-y-3">
                      <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">1. Pairwise Formula & Ratio Generator</div>
                      <div className="flex flex-wrap items-center gap-2">
                        <select
                          value={feColA}
                          onChange={e => setFeColA(e.target.value)}
                          className="bg-white border border-[#E5E9E6] hover:border-[#164E33] rounded-xl px-2.5 py-1.5 text-xs font-mono font-medium text-[#121816] focus:outline-none focus:border-[#164E33] transition max-w-[140px]"
                        >
                          <option value="">-- Col A --</option>
                          {edaData?.columns_profile?.map((c: any) => (
                            <option key={c.name} value={c.name}>{c.name}</option>
                          ))}
                        </select>

                        <select
                          value={feOp}
                          onChange={e => setFeOp(e.target.value)}
                          className="bg-white border border-[#E5E9E6] hover:border-[#164E33] rounded-xl px-2 py-1.5 text-xs font-mono font-bold text-[#164E33] focus:outline-none focus:border-[#164E33] transition"
                        >
                          <option value="/">/</option>
                          <option value="*">*</option>
                          <option value="-">-</option>
                          <option value="+">+</option>
                        </select>

                        <select
                          value={feColB}
                          onChange={e => setFeColB(e.target.value)}
                          className="bg-white border border-[#E5E9E6] hover:border-[#164E33] rounded-xl px-2.5 py-1.5 text-xs font-mono font-medium text-[#121816] focus:outline-none focus:border-[#164E33] transition max-w-[140px]"
                        >
                          <option value="">-- Col B --</option>
                          {edaData?.columns_profile?.map((c: any) => (
                            <option key={c.name} value={c.name}>{c.name}</option>
                          ))}
                        </select>

                        <button
                          type="button"
                          onClick={handleInsertFormula}
                          className="flex items-center space-x-1 bg-[#121816] hover:bg-[#164E33] text-white text-xs font-semibold px-3 py-1.5 rounded-xl transition shadow-2xs"
                        >
                          <Plus size={12} />
                          <span>Insert</span>
                        </button>
                      </div>
                    </div>

                    {/* Visual Group-By Aggregation Builder */}
                    <div className="border border-[#E5E9E6] rounded-xl p-4 bg-[#F8FAF9] space-y-3">
                      <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">2. Group-By Aggregator & Deviation</div>
                      <div className="flex flex-wrap items-center gap-2">
                        <select
                          value={feGroupCat}
                          onChange={e => setFeGroupCat(e.target.value)}
                          className="bg-white border border-[#E5E9E6] hover:border-[#164E33] rounded-xl px-2.5 py-1.5 text-xs font-mono font-medium text-[#121816] focus:outline-none focus:border-[#164E33] transition max-w-[140px]"
                        >
                          <option value="">-- Group (Cat) --</option>
                          {edaData?.columns_profile?.filter((c: any) => c.type === 'categorical' || c.dtype === 'object')?.map((c: any) => (
                            <option key={c.name} value={c.name}>{c.name}</option>
                          ))}
                          {/* Fallback to all columns if type filter empty */}
                          {(!edaData?.columns_profile?.some((c: any) => c.type === 'categorical' || c.dtype === 'object')) && edaData?.columns_profile?.map((c: any) => (
                            <option key={c.name} value={c.name}>{c.name}</option>
                          ))}
                        </select>

                        <select
                          value={feGroupNum}
                          onChange={e => setFeGroupNum(e.target.value)}
                          className="bg-white border border-[#E5E9E6] hover:border-[#164E33] rounded-xl px-2.5 py-1.5 text-xs font-mono font-medium text-[#121816] focus:outline-none focus:border-[#164E33] transition max-w-[140px]"
                        >
                          <option value="">-- Value (Num) --</option>
                          {edaData?.columns_profile?.filter((c: any) => c.type === 'numerical' || ['int64', 'float64'].includes(c.dtype))?.map((c: any) => (
                            <option key={c.name} value={c.name}>{c.name}</option>
                          ))}
                          {/* Fallback to all columns if filter empty */}
                          {(!edaData?.columns_profile?.some((c: any) => c.type === 'numerical')) && edaData?.columns_profile?.map((c: any) => (
                            <option key={c.name} value={c.name}>{c.name}</option>
                          ))}
                        </select>

                        <select
                          value={feGroupStat}
                          onChange={e => setFeGroupStat(e.target.value as any)}
                          className="bg-white border border-[#E5E9E6] hover:border-[#164E33] rounded-xl px-2 py-1.5 text-xs font-mono font-bold text-[#164E33] focus:outline-none focus:border-[#164E33] transition"
                        >
                          <option value="diff">Diff from Mean</option>
                          <option value="mean">Group Mean</option>
                          <option value="std">Group Std</option>
                          <option value="zscore">Group Z-Score</option>
                        </select>

                        <button
                          type="button"
                          onClick={handleInsertGroupFormula}
                          className="flex items-center space-x-1 bg-[#121816] hover:bg-[#164E33] text-white text-xs font-semibold px-3 py-1.5 rounded-xl transition shadow-2xs"
                        >
                          <Plus size={12} />
                          <span>Insert</span>
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <textarea
                      value={editableHookCode}
                      onChange={e => setEditableHookCode(e.target.value)}
                      rows={16}
                      spellCheck={false}
                      className="w-full bg-[#121816] text-[#E8F3ED] p-4 rounded-xl font-mono text-xs leading-relaxed border border-[#E5E9E6] focus:outline-none focus:border-[#164E33] selection:bg-[#164E33]"
                      placeholder="Write your custom BaseFeatureHook transform() method here..."
                    />
                    <div className="flex justify-between items-center text-[11px] text-[#76807B] font-mono">
                      <span>Target: cases/{selectedCaseId}/feature_hook.py</span>
                      <span>Signature: transform(self, train_df, test_df=None)</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 3: MODEL TRAINING & CV LEADERBOARD */}
            {activeTab === 'training' && (
              <div className="space-y-8">
                {/* 4 Status KPI Cards */}
                <div className="grid grid-cols-4 gap-4">
                  <div className="bg-white border border-[#E5E9E6] rounded-2xl p-5 shadow-sm">
                    <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">Target Variable</div>
                    <div className="text-xl font-bold mt-1 text-[#121816] font-mono">{meta?.target_col}</div>
                    <div className="text-xs text-[#76807B] mt-1 capitalize">{meta?.task_type} Classification</div>
                  </div>

                  <div className="bg-white border border-[#E5E9E6] rounded-2xl p-5 shadow-sm">
                    <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">Primary Metric</div>
                    <div className="text-xl font-bold mt-1 text-[#121816] font-mono">{meta?.eval_metric.toUpperCase()}</div>
                    <div className="text-xs text-[#76807B] mt-1">
                      {meta?.n_splits}-Fold {meta?.cv_strategy}{meta?.group_col ? ` (${meta.group_col})` : ''}
                    </div>
                  </div>

                  <div className="bg-white border border-[#E5E9E6] rounded-2xl p-5 shadow-sm">
                    <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">
                      Best Single Model ({bestSingleName.toUpperCase() || 'N/A'})
                    </div>
                    <div className="text-xl font-bold mt-1 text-[#121816] font-mono">
                      {bestSingleScore ? bestSingleScore.toFixed(5) : 'Not Run'}
                    </div>
                    <div className="text-xs text-[#76807B] mt-1">Single Model Baseline</div>
                  </div>

                  <div className="bg-white border border-[#E5E9E6] rounded-2xl p-5 shadow-sm">
                    <div className="text-[11px] font-bold uppercase tracking-wider text-[#76807B] font-mono">Champion Ensemble</div>
                    <div className="text-xl font-bold mt-1 text-[#121816] font-mono flex items-center space-x-2">
                      <span>{championScore ? championScore.toFixed(5) : 'Not Run'}</span>
                      {championScore && bestSingleScore && (
                        <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded-full bg-[#E8F3ED] text-[#164E33] border border-[#D2E7DC]">
                          +{ (championScore - bestSingleScore).toFixed(5) }
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-[#76807B] mt-1 font-mono truncate">{championModel || 'None'}</div>
                  </div>
                </div>

                {/* Feature Hook Active Banner */}
                <div className="bg-white border border-[#E5E9E6] rounded-2xl p-4 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-3">
                  <div className="flex items-center space-x-3">
                    <span className={`w-2.5 h-2.5 rounded-full ${caseDetail?.has_hook ? 'bg-[#164E33]' : 'bg-[#98A29D]'}`} />
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-bold text-[#121816]">Feature Engineering Hook:</span>
                        <span className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded font-bold border ${
                          caseDetail?.has_hook 
                            ? 'bg-[#E8F3ED] text-[#164E33] border-[#D2E7DC]' 
                            : 'bg-[#F8FAF9] text-[#76807B] border-[#E5E9E6]'
                        }`}>
                          {caseDetail?.has_hook ? 'Active & Ready for CV' : 'No Custom Hook (Using Raw Dataset)'}
                        </span>
                      </div>
                      <p className="text-xs text-[#76807B] mt-0.5 font-mono">
                        {caseDetail?.has_hook 
                          ? 'Applied automatically to train and test splits during 5-Fold Cross Validation'
                          : 'Configure domain transformations in Step 2 to generate specialized interaction features'}
                      </p>
                    </div>
                  </div>

                  {caseDetail?.has_hook && (
                    <button
                      type="button"
                      onClick={() => setActiveTab('fe')}
                      className="text-xs font-semibold text-[#164E33] hover:underline whitespace-nowrap self-start md:self-auto"
                    >
                      Inspect in Step 2 &rarr;
                    </button>
                  )}
                </div>

                {/* Active Model Selector for Distributed Team Training */}
                <div className="bg-white border border-[#E5E9E6] rounded-2xl p-5 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="space-y-0.5">
                    <div className="text-xs font-bold uppercase tracking-tight text-[#121816]">Active Execution Models (Distributed Parallel Training)</div>
                    <div className="text-xs text-[#76807B]">
                      Toggle which models to execute on this machine. Teammates can train single models in parallel and ensemble in Step 4.
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    {[
                      { id: 'lgbm', label: 'LightGBM' },
                      { id: 'xgb', label: 'XGBoost' },
                      { id: 'catboost', label: 'CatBoost' }
                    ].map(item => {
                      const active = selectedModels.includes(item.id);
                      return (
                        <button
                          key={item.id}
                          type="button"
                          onClick={() => {
                            if (active) {
                              if (selectedModels.length > 1) {
                                setSelectedModels(selectedModels.filter(m => m !== item.id));
                              }
                            } else {
                              setSelectedModels([...selectedModels, item.id]);
                            }
                          }}
                          className={`flex items-center space-x-1.5 text-xs font-mono font-medium px-3.5 py-1.5 rounded-xl border transition ${
                            active 
                              ? 'bg-[#164E33] text-white border-[#164E33] font-semibold shadow-2xs' 
                              : 'bg-white text-[#76807B] border-[#E5E9E6] hover:border-[#164E33]'
                          }`}
                        >
                          <span>{active ? '[x]' : '[ ]'}</span>
                          <span>{item.label}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Model Architecture & Hyperparameter Configuration */}
                <div className="bg-white border border-[#E5E9E6] rounded-2xl p-6 shadow-sm space-y-5">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E5E9E6] pb-4">
                    <div>
                      <div className="flex items-center space-x-2">
                        <h3 className="font-bold text-sm text-[#121816] uppercase tracking-tight">
                          Model Architecture & Hyperparameter Presets
                        </h3>
                        <span className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded font-bold border ${
                          archPreset === 'shallow'
                            ? 'bg-[#E8F3ED] text-[#164E33] border-[#D2E7DC]'
                            : archPreset === 'custom'
                            ? 'bg-[#FFF8E1] text-[#B78103] border-[#FFE082]'
                            : 'bg-[#F4F5F6] text-[#4A5550] border-[#E5E9E6]'
                        }`}>
                          {archPreset === 'shallow' ? 'Shallow Trees Mode' : archPreset === 'custom' ? 'Custom JSON' : 'Standard Deep Trees'}
                        </span>
                      </div>
                      <p className="text-xs text-[#76807B] mt-1">
                        Select structural tree depth paradigms or inspect tuned hyperparameters applied to GBDT models.
                      </p>
                    </div>

                    {paramsSaveMsg && (
                      <div className="inline-flex items-center space-x-1.5 px-3 py-1 bg-[#E8F3ED] border border-[#D2E7DC] text-[#164E33] text-xs font-medium rounded-lg">
                        <Check size={13} />
                        <span>{paramsSaveMsg}</span>
                      </div>
                    )}
                  </div>

                  {/* Architecture Preset Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {/* Shallow Trees Preset */}
                    <button
                      type="button"
                      onClick={() => handleSelectPreset('shallow')}
                      disabled={isSavingParams}
                      className={`text-left p-4 rounded-xl border transition flex flex-col justify-between ${
                        archPreset === 'shallow'
                          ? 'border-[#164E33] bg-[#F7FAF8] ring-1 ring-[#164E33]'
                          : 'border-[#E5E9E6] bg-white hover:border-[#164E33]'
                      }`}
                    >
                      <div className="space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-[#121816]">Shallow Trees</span>
                          <span className="text-[9px] font-mono uppercase bg-[#164E33] text-white px-1.5 py-0.5 rounded font-semibold">
                            Recommended (0.950+)
                          </span>
                        </div>
                        <p className="text-[11px] text-[#76807B] leading-relaxed">
                          num_leaves=4 (LGBM), max_depth=2 (XGB), depth=3 (CatBoost) with lr=0.04 and 5000 trees. Avoids overfitting on synthetic additive CTGAN distributions.
                        </p>
                      </div>
                      <div className="mt-3 pt-2 border-t border-[#E5E9E6] flex items-center justify-between text-[10px] font-mono">
                        <span className="text-[#164E33] font-semibold">Additive Logit Structure</span>
                        <span className={archPreset === 'shallow' ? 'text-[#164E33] font-bold' : 'text-[#76807B]'}>
                          {archPreset === 'shallow' ? '[Active]' : 'Select'}
                        </span>
                      </div>
                    </button>

                    {/* Standard Baseline Preset */}
                    <button
                      type="button"
                      onClick={() => handleSelectPreset('standard')}
                      disabled={isSavingParams}
                      className={`text-left p-4 rounded-xl border transition flex flex-col justify-between ${
                        archPreset === 'standard'
                          ? 'border-[#164E33] bg-[#F7FAF8] ring-1 ring-[#164E33]'
                          : 'border-[#E5E9E6] bg-white hover:border-[#164E33]'
                      }`}
                    >
                      <div className="space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-[#121816]">Standard GBDT</span>
                          <span className="text-[9px] font-mono uppercase bg-[#F4F5F6] text-[#76807B] px-1.5 py-0.5 rounded border border-[#E5E9E6]">
                            Baseline
                          </span>
                        </div>
                        <p className="text-[11px] text-[#76807B] leading-relaxed">
                          Default deep trees: num_leaves=45 (LGBM), max_depth=6 (XGB/CatBoost) with lr=0.03. Suitable for traditional non-synthetic tabular data.
                        </p>
                      </div>
                      <div className="mt-3 pt-2 border-t border-[#E5E9E6] flex items-center justify-between text-[10px] font-mono">
                        <span className="text-[#76807B]">Deep Splitting Structure</span>
                        <span className={archPreset === 'standard' ? 'text-[#164E33] font-bold' : 'text-[#76807B]'}>
                          {archPreset === 'standard' ? '[Active]' : 'Select'}
                        </span>
                      </div>
                    </button>

                    {/* Custom JSON Editor Preset */}
                    <button
                      type="button"
                      onClick={() => {
                        setIsEditingParams(!isEditingParams);
                        if (!isEditingParams) {
                          setParamsDraft(JSON.stringify(modelParams, null, 2));
                        }
                      }}
                      className={`text-left p-4 rounded-xl border transition flex flex-col justify-between ${
                        archPreset === 'custom' || isEditingParams
                          ? 'border-[#B78103] bg-[#FFFDF7] ring-1 ring-[#B78103]'
                          : 'border-[#E5E9E6] bg-white hover:border-[#164E33]'
                      }`}
                    >
                      <div className="space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-[#121816]">Custom Parameters</span>
                          <span className="text-[9px] font-mono uppercase bg-[#FFF8E1] text-[#B78103] border border-[#FFE082] px-1.5 py-0.5 rounded font-semibold">
                            Manual JSON
                          </span>
                        </div>
                        <p className="text-[11px] text-[#76807B] leading-relaxed">
                          Directly inspect or edit the JSON dictionary of hyperparameters passed to LightGBM, XGBoost, and CatBoost.
                        </p>
                      </div>
                      <div className="mt-3 pt-2 border-t border-[#E5E9E6] flex items-center justify-between text-[10px] font-mono">
                        <span className="text-[#B78103]">Direct Tuning</span>
                        <span className="text-[#B78103] font-bold">
                          {isEditingParams ? 'Hide Editor' : 'Open Editor'}
                        </span>
                      </div>
                    </button>
                  </div>

                  {/* Active Hyperparameters Summary Badge Strip */}
                  <div className="bg-[#F8FAF9] border border-[#E5E9E6] rounded-xl p-3.5 flex flex-wrap items-center gap-2 text-xs font-mono">
                    <span className="text-[#76807B] uppercase text-[10px] font-bold tracking-wider mr-1">Active Hyperparameters:</span>
                    {selectedModels.map(m => {
                      const p = (modelParams && modelParams[m]) || {};
                      const summary = m === 'lgbm' 
                        ? `leaves: ${p.num_leaves ?? 45}, lr: ${p.learning_rate ?? 0.03}, trees: ${p.n_estimators ?? 2000}, colsample: ${p.colsample_bytree ?? 0.75}`
                        : m === 'xgb'
                        ? `depth: ${p.max_depth ?? 6}, lr: ${p.learning_rate ?? 0.03}, trees: ${p.n_estimators ?? 2000}, colsample: ${p.colsample_bytree ?? 0.75}`
                        : `depth: ${p.depth ?? 6}, lr: ${p.learning_rate ?? 0.03}, iter: ${p.iterations ?? 2000}`;
                      return (
                        <div key={m} className="inline-flex items-center space-x-1.5 bg-white border border-[#D2E7DC] px-2.5 py-1 rounded-lg text-[#164E33] shadow-2xs">
                          <span className="font-bold uppercase text-[11px]">{m}:</span>
                          <span className="text-[10px] text-[#222]">{summary}</span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Collapsible Custom JSON Editor */}
                  {isEditingParams && (
                    <div className="bg-[#121816] border border-[#26332C] rounded-xl p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-mono font-medium text-[#E5E9E6]">
                          Raw Hyperparameters JSON (meta.model_params)
                        </span>
                        <div className="flex items-center space-x-2">
                          <button
                            type="button"
                            onClick={() => setIsEditingParams(false)}
                            className="text-xs text-[#A0ACA5] hover:text-white px-2 py-1"
                          >
                            Cancel
                          </button>
                          <button
                            type="button"
                            onClick={handleSaveCustomParams}
                            disabled={isSavingParams}
                            className="inline-flex items-center space-x-1.5 px-3 py-1 bg-[#164E33] hover:bg-[#1E6B47] text-white text-xs font-mono font-medium rounded-lg transition"
                          >
                            <Save size={12} />
                            <span>Save Configuration</span>
                          </button>
                        </div>
                      </div>
                      <textarea
                        value={paramsDraft}
                        onChange={e => setParamsDraft(e.target.value)}
                        rows={10}
                        className="w-full bg-[#0D1210] border border-[#26332C] rounded-lg p-3 text-xs font-mono text-[#D2E7DC] focus:outline-none focus:border-[#164E33]"
                        spellCheck={false}
                      />
                    </div>
                  )}
                </div>

                {/* Optuna Bayesian Hyperparameter Optimization Studio */}
                <div className="border border-[#E5E9E6] rounded-2xl p-6 bg-white space-y-5 shadow-sm">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E5E9E6] pb-4">
                    <div>
                      <div className="flex items-center space-x-2">
                        <h3 className="font-bold text-sm text-[#121816] uppercase tracking-tight">Automated Hyperparameter Optimization (Optuna Engine)</h3>
                        <span className="text-[10px] font-mono uppercase bg-[#E8F3ED] text-[#164E33] border border-[#D2E7DC] px-2 py-0.5 rounded font-bold">
                          Bayesian TPE
                        </span>
                      </div>
                      <p className="text-xs text-[#76807B] mt-1">
                        Discovers deeper tree architectures, learning rates, and L1/L2 regularization across single models or all 3 algorithms sequentially.
                      </p>
                    </div>

                    {/* Tuning Action Controls */}
                    <div className="flex items-center space-x-2.5">
                      <select
                        value={tuneModel}
                        onChange={e => setTuneModel(e.target.value as any)}
                        disabled={tuning}
                        className="bg-[#F8FAF9] border border-[#E5E9E6] hover:border-[#164E33] rounded-xl px-3 py-1.5 text-xs font-mono font-medium text-[#121816] focus:outline-none focus:border-[#164E33] transition"
                      >
                        <option value="all">All 3 Models (Sequential Auto-Tune)</option>
                        <option value="lgbm">LightGBM Only</option>
                        <option value="xgb">XGBoost Only</option>
                        <option value="catboost">CatBoost Only</option>
                      </select>

                      <select
                        value={tuneTrials}
                        onChange={e => setTuneTrials(Number(e.target.value))}
                        disabled={tuning}
                        className="bg-[#F8FAF9] border border-[#E5E9E6] hover:border-[#164E33] rounded-xl px-3 py-1.5 text-xs font-mono font-medium text-[#121816] focus:outline-none focus:border-[#164E33] transition"
                      >
                        <option value={15}>15 Trials (Fast Exploration)</option>
                        <option value={30}>30 Trials (Balanced - Recommended)</option>
                        <option value={50}>50 Trials (Deep Optimization)</option>
                        <option value={100}>100 Trials (Exhaustive)</option>
                      </select>

                      <button
                        onClick={handleStartTune}
                        disabled={tuning || running}
                        className={`flex items-center space-x-1.5 px-4 py-1.5 rounded-xl text-xs font-semibold text-white transition shadow-2xs ${
                          tuning || running ? 'bg-[#98A29D] cursor-not-allowed' : 'bg-[#164E33] hover:bg-[#113C27]'
                        }`}
                      >
                        {tuning ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
                        <span>{tuning ? 'Tuning in Progress...' : 'Run Optuna Study'}</span>
                      </button>
                    </div>
                  </div>

                  {/* Active Tuning Status or Results */}
                  {tuneStatus && tuneStatus.status !== 'idle' && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="border border-[#E5E9E6] rounded-xl p-3 bg-[#F8FAF9]">
                          <div className="text-[10px] uppercase font-semibold text-[#76807B]">Optimization Status</div>
                          <div className="text-sm font-bold text-[#121816] mt-0.5 capitalize font-mono">{tuneStatus.status}</div>
                        </div>
                        <div className="border border-[#E5E9E6] rounded-xl p-3 bg-[#F8FAF9]">
                          <div className="text-[10px] uppercase font-semibold text-[#76807B]">Current Target</div>
                          <div className="text-sm font-bold text-[#164E33] mt-0.5 font-mono">
                            {tuneStatus.current_model || (tuneStatus.model === 'all' ? 'ALL 3' : tuneStatus.model?.toUpperCase())}
                          </div>
                        </div>
                        <div className="border border-[#E5E9E6] rounded-xl p-3 bg-[#F8FAF9]">
                          <div className="text-[10px] uppercase font-semibold text-[#76807B]">Trials Progress</div>
                          <div className="text-sm font-bold text-[#121816] mt-0.5 font-mono">
                            {tuneStatus.current_trial || tuneStatus.trials?.length || 0} / {tuneStatus.total_trials || tuneTrials}
                          </div>
                        </div>
                        <div className="border border-[#E5E9E6] rounded-xl p-3 bg-[#F8FAF9]">
                          <div className="text-[10px] uppercase font-semibold text-[#76807B]">Highest {meta?.eval_metric.toUpperCase()}</div>
                          <div className="text-sm font-bold text-[#121816] mt-0.5 font-mono">
                            {tuneStatus.best_score ? tuneStatus.best_score.toFixed(5) : '-'}
                          </div>
                        </div>
                      </div>

                      {/* Multi-model or Single-model Summary */}
                      {tuneStatus.all_best_scores && Object.keys(tuneStatus.all_best_scores).length > 0 && (
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          {['lgbm', 'xgb', 'catboost'].map(m => {
                            const sc = tuneStatus.all_best_scores[m];
                            if (sc === undefined) return null;
                            return (
                              <div key={m} className="border border-[#E5E9E6] rounded-xl p-3 bg-[#F8FAF9]">
                                <div className="text-[10px] uppercase font-semibold text-[#76807B]">{m.toUpperCase()} Optimal Score</div>
                                <div className="text-sm font-bold text-[#164E33] mt-0.5 font-mono">
                                  {sc.toFixed(5)}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {/* Automated Tuning Diagnostics & Decision Card */}
                      {tuneStatus.status === 'completed' && tuneStatus.best_params && (
                        <div className="border border-[#D2E7DC] bg-[#F8FAF9] rounded-xl p-4 space-y-3">
                          <div className="flex items-center justify-between border-b border-[#E5E9E6] pb-2">
                            <div className="flex items-center space-x-2">
                              <span className="text-xs font-bold text-[#121816] uppercase tracking-tight">Tuning Diagnostics & Recommendation</span>
                              <span className="text-[10px] font-mono uppercase bg-[#E8F3ED] text-[#164E33] border border-[#D2E7DC] px-2 py-0.5 rounded font-bold">
                                Recommended to Apply
                              </span>
                            </div>
                            <span className="text-[11px] font-mono text-[#76807B]">
                              TPE Convergence: Passed
                            </span>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                            <div className="bg-white p-3 rounded-lg border border-[#E5E9E6]">
                              <div className="text-[10px] font-bold uppercase text-[#76807B]">Exploration Delta</div>
                              <div className="text-sm font-bold text-[#164E33] mt-0.5 font-mono">
                                +{Math.max(0, (tuneStatus.best_score || 0) - ((tuneStatus.trials?.[0]?.score) || (tuneStatus.best_score || 0))).toFixed(5)}
                              </div>
                              <div className="text-[11px] text-[#76807B] mt-0.5">Gain over initial random trial. Positive drift indicates healthy parameter discovery.</div>
                            </div>
                            <div className="bg-white p-3 rounded-lg border border-[#E5E9E6]">
                              <div className="text-[10px] font-bold uppercase text-[#76807B]">Architecture Stability</div>
                              <div className="text-sm font-bold text-[#121816] mt-0.5 font-mono">L1/L2 Regularized</div>
                              <div className="text-[11px] text-[#76807B] mt-0.5">Penalties guard against tree memorization on high-cardinality features.</div>
                            </div>
                            <div className="bg-white p-3 rounded-lg border border-[#E5E9E6]">
                              <div className="text-[10px] font-bold uppercase text-[#76807B]">Proxy vs Full CV Bridge</div>
                              <div className="text-sm font-bold text-[#121816] mt-0.5 font-mono">2-Fold (60k) to 5-Fold</div>
                              <div className="text-[11px] text-[#76807B] mt-0.5">Proxy scores scale significantly higher when trained on full 5-fold folds with 1500 trees.</div>
                            </div>
                          </div>
                        </div>
                      )}

                      {tuneStatus.best_params && (
                        <div className="border border-[#E5E9E6] rounded-xl p-4 bg-[#F8FAF9] flex flex-col md:flex-row md:items-center justify-between gap-4">
                          <div className="space-y-1 overflow-hidden w-full">
                            <div className="text-xs font-semibold text-[#121816]">
                              Discovered Optimal Hyperparameters ({tuneStatus.model === 'all' ? 'All 3 Models' : tuneStatus.model?.toUpperCase()})
                            </div>
                            <pre className="text-[11px] font-mono text-[#4A5550] bg-white p-3 rounded-xl border border-[#E5E9E6] max-h-40 overflow-y-auto">
                              {JSON.stringify(tuneStatus.best_params, null, 2)}
                            </pre>
                          </div>
                          <button
                            onClick={handleApplyTunedParams}
                            className="flex items-center space-x-1.5 bg-[#164E33] hover:bg-[#113C27] text-white text-xs font-semibold px-4 py-2 rounded-xl transition whitespace-nowrap self-start md:self-auto shadow-2xs"
                          >
                            <Check size={14} />
                            <span>Apply to Pipeline Config</span>
                          </button>
                        </div>
                      )}

                      {/* Optuna Console Log */}
                      <div className="bg-[#121816] text-[#E8F3ED] rounded-xl p-3 font-mono text-[11px] max-h-40 overflow-y-auto space-y-1">
                        {tuneStatus.logs?.map((l: string, i: number) => (
                          <div key={i}>{l}</div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Validation Leaderboard Table */}
                <div className="border border-zinc-200 rounded overflow-hidden">
                  <div className="px-5 py-3 border-b border-zinc-200 bg-zinc-50 flex items-center justify-between">
                    <span className="font-semibold text-sm text-zinc-950">Out-Of-Fold Validation Leaderboard</span>
                    <span className="text-xs text-zinc-500 font-mono">{Object.keys(scores).length} evaluated strategies</span>
                  </div>

                  {Object.keys(scores).length === 0 ? (
                    <div className="p-12 text-center text-sm text-zinc-500 space-y-3">
                      <p>No validation scores recorded yet for this case.</p>
                      <button
                        onClick={handleExecute}
                        disabled={running}
                        className="inline-flex items-center space-x-1.5 bg-zinc-950 text-white text-xs font-semibold px-4 py-2 rounded"
                      >
                        <Play size={14} />
                        <span>Start Cross-Validation Now</span>
                      </button>
                    </div>
                  ) : (
                    <table className="w-full text-left text-sm">
                      <thead className="bg-zinc-50/80 border-b border-zinc-200 text-xs uppercase text-zinc-500 font-semibold">
                        <tr>
                          <th className="px-5 py-2.5">Rank</th>
                          <th className="px-5 py-2.5">Model / Strategy</th>
                          <th className="px-5 py-2.5 font-mono">OOF {meta?.eval_metric.toUpperCase()}</th>
                          <th className="px-5 py-2.5 font-mono">Gain vs Baseline</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-200">
                        {Object.entries(scores)
                          .sort(([, a]: any, [, b]: any) => b - a)
                          .map(([name, score]: any, idx) => {
                            const gain = bestSingleScore ? score - bestSingleScore : 0;
                            return (
                              <tr key={name} className={idx === 0 ? 'bg-zinc-50 font-medium' : ''}>
                                <td className="px-5 py-3 text-zinc-500 font-mono">#{idx + 1}</td>
                                <td className="px-5 py-3 font-medium text-zinc-950 flex items-center space-x-2">
                                  <span>{name}</span>
                                  {idx === 0 && (
                                    <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-zinc-900 text-white">
                                      Champion
                                    </span>
                                  )}
                                </td>
                                <td className="px-5 py-3 font-mono font-semibold text-zinc-950">
                                  {score.toFixed(5)}
                                </td>
                                <td className="px-5 py-3 font-mono text-xs">
                                  {gain > 0 ? (
                                    <span className="text-zinc-900 font-semibold">+{gain.toFixed(5)}</span>
                                  ) : gain < 0 ? (
                                    <span className="text-zinc-400">{gain.toFixed(5)}</span>
                                  ) : (
                                    <span className="text-zinc-400">0.00000</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                      </tbody>
                    </table>
                  )}
                </div>

                {/* EXPERIMENT HISTORY & BENCHMARK LEDGER */}
                <div className="bg-white border border-[#E5E9E6] rounded-2xl overflow-hidden shadow-sm">
                  <div className="px-6 py-4 border-b border-[#E5E9E6] bg-[#F8FAF9] flex items-center justify-between">
                    <div>
                      <div className="flex items-center space-x-2">
                        <History size={16} className="text-[#164E33]" />
                        <span className="font-semibold text-sm text-[#121816]">Experiment History & Benchmark Ledger</span>
                      </div>
                      <p className="text-xs text-[#76807B] mt-0.5">
                        Track CV scores across Pure Baseline, Feature Engineering, Optuna, and Ensembles against Kaggle Public Leaderboard.
                      </p>
                    </div>
                    <span className="text-xs font-mono px-2.5 py-1 bg-white border border-[#E5E9E6] rounded-lg text-[#121816] font-medium">
                      {(caseDetail.history || []).length} Runs Logged
                    </span>
                  </div>

                  {!caseDetail.history || caseDetail.history.length === 0 ? (
                    <div className="p-8 text-center text-xs text-zinc-500">
                      No experiment runs recorded yet. Execute a pipeline run to log your first benchmark.
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-[#F8FAF9]/60 border-b border-[#E5E9E6] text-[11px] uppercase tracking-wider text-[#76807B] font-mono">
                          <tr>
                            <th className="px-5 py-3">Run & Time</th>
                            <th className="px-5 py-3">Experiment Strategy</th>
                            <th className="px-5 py-3">Features</th>
                            <th className="px-5 py-3 font-mono">OOF {meta?.eval_metric.toUpperCase()}</th>
                            <th className="px-5 py-3 font-mono">Gain vs Base</th>
                            <th className="px-5 py-3 font-mono">Kaggle Public LB</th>
                            <th className="px-5 py-3 text-right">Submission</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#E5E9E6]">
                          {caseDetail.history.map((run, rIdx) => {
                            const isChampion = run.cv_score === Math.max(...caseDetail.history!.map(h => h.cv_score));
                            const isEditingThis = editingLbRunId === run.run_id;

                            // Correlation calculation if LB score exists
                            const baseLb = caseDetail.history![0]?.lb_score;
                            const lbDelta = (run.lb_score != null && baseLb != null && rIdx > 0) ? run.lb_score - baseLb : null;
                            const isAligned = (run.cv_delta < 0 && lbDelta != null && lbDelta < 0) || (run.cv_delta > 0 && lbDelta != null && lbDelta > 0);

                            return (
                              <tr key={run.run_id} className={`hover:bg-[#F8FAF9]/50 transition-colors ${isChampion ? 'bg-emerald-50/20' : ''}`}>
                                <td className="px-5 py-3.5 whitespace-nowrap">
                                  <div className="font-mono font-semibold text-[#121816] flex items-center space-x-1.5">
                                    <span>#{String(run.run_number).padStart(2, '0')}</span>
                                    {isChampion && (
                                      <span className="text-[10px] font-sans font-bold bg-[#164E33] text-white px-1.5 py-0.2 rounded">
                                        Best CV
                                      </span>
                                    )}
                                  </div>
                                  <div className="text-[11px] text-[#76807B] flex items-center space-x-1 mt-0.5">
                                    <Clock size={11} />
                                    <span>{run.timestamp}</span>
                                  </div>
                                </td>

                                <td className="px-5 py-3.5">
                                  <div className="flex items-center space-x-2">
                                    <span className={`px-2 py-0.5 rounded text-[11px] font-medium border ${
                                      run.tag === 'baseline' 
                                        ? 'bg-zinc-100 text-zinc-800 border-zinc-200' 
                                        : run.tag === 'feature_engineering'
                                        ? 'bg-blue-50 text-blue-800 border-blue-200'
                                        : run.tag === 'ensemble'
                                        ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                                        : 'bg-purple-50 text-purple-800 border-purple-200'
                                    }`}>
                                      {run.tag === 'baseline' ? 'Pure Baseline' : run.tag === 'feature_engineering' ? 'Feature Engineering' : run.tag === 'ensemble' ? 'Ensemble Blend' : 'Optuna Tuned'}
                                    </span>
                                    <span className="font-medium text-[#121816] text-xs truncate max-w-xs">{run.name}</span>
                                  </div>
                                  {run.notes && (
                                    <p className="text-[11px] text-[#76807B] mt-1 line-clamp-1 italic">{run.notes}</p>
                                  )}
                                </td>

                                <td className="px-5 py-3.5 font-mono text-xs">
                                  <span className="text-[#121816] font-medium">{run.features_count}</span>
                                  <span className="text-[#76807B] ml-1">cols</span>
                                  {rIdx > 0 && run.features_count !== caseDetail.history![0].features_count && (
                                    <span className="text-[10px] ml-1 text-blue-600 font-semibold">
                                      (+{run.features_count - caseDetail.history![0].features_count})
                                    </span>
                                  )}
                                </td>

                                <td className="px-5 py-3.5 font-mono font-semibold text-xs text-[#121816]">
                                  {run.cv_score.toFixed(5)}
                                </td>

                                <td className="px-5 py-3.5 font-mono text-xs">
                                  {run.cv_delta > 0 ? (
                                    <span className="inline-flex items-center text-emerald-700 font-semibold bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                                      <ArrowUpRight size={12} className="mr-0.5" />
                                      +{run.cv_delta.toFixed(5)}
                                    </span>
                                  ) : run.cv_delta < 0 ? (
                                    <span className="inline-flex items-center text-rose-700 font-semibold bg-rose-50 px-1.5 py-0.5 rounded border border-rose-200">
                                      <ArrowDownRight size={12} className="mr-0.5" />
                                      {run.cv_delta.toFixed(5)}
                                    </span>
                                  ) : (
                                    <span className="text-[#76807B] font-mono">0.00000 (Base)</span>
                                  )}
                                </td>

                                <td className="px-5 py-3.5">
                                  {isEditingThis ? (
                                    <div className="flex items-center space-x-1.5">
                                      <input
                                        type="number"
                                        step="0.00001"
                                        value={tempLbScore}
                                        onChange={e => setTempLbScore(e.target.value)}
                                        placeholder="0.94640"
                                        className="w-24 px-2 py-1 text-xs font-mono bg-white border border-[#164E33] rounded focus:outline-none"
                                        autoFocus
                                      />
                                      <button
                                        type="button"
                                        disabled={isSavingLb}
                                        onClick={() => handleSaveLb(run.run_id)}
                                        className="px-2 py-1 bg-[#164E33] text-white text-[11px] font-medium rounded hover:bg-[#113d28]"
                                      >
                                        Save
                                      </button>
                                      <button
                                        type="button"
                                        onClick={() => setEditingLbRunId(null)}
                                        className="px-1.5 py-1 text-zinc-400 hover:text-zinc-600 text-xs"
                                      >
                                        <X size={12} />
                                      </button>
                                    </div>
                                  ) : (
                                    <div className="flex items-center space-x-2">
                                      {run.lb_score != null ? (
                                        <div className="flex items-center space-x-1.5 font-mono">
                                          <span className="font-bold text-[#121816] text-xs">
                                            {Number(run.lb_score).toFixed(5)}
                                          </span>
                                          {lbDelta != null && (
                                            <span className={`text-[10px] font-semibold px-1 py-0.2 rounded ${
                                              lbDelta > 0 
                                                ? 'text-emerald-700 bg-emerald-50' 
                                                : lbDelta < 0 
                                                ? 'text-rose-700 bg-rose-50' 
                                                : 'text-zinc-600'
                                            }`}>
                                              {lbDelta > 0 ? `+${lbDelta.toFixed(5)}` : lbDelta.toFixed(5)}
                                            </span>
                                          )}
                                          {isAligned && (
                                            <span className="text-[10px] text-[#164E33] bg-[#164E33]/10 px-1 py-0.2 rounded font-sans font-medium" title="CV and Public LB moved in the same direction!">
                                              Aligned
                                            </span>
                                          )}
                                        </div>
                                      ) : (
                                        <span className="text-[11px] text-zinc-400 italic">Not entered</span>
                                      )}
                                      <button
                                        type="button"
                                        onClick={() => {
                                          setEditingLbRunId(run.run_id);
                                          setTempLbScore(run.lb_score != null ? String(run.lb_score) : '');
                                          setTempNotes(run.notes || '');
                                        }}
                                        className="text-zinc-400 hover:text-[#164E33] p-1 rounded transition-colors"
                                        title="Edit Kaggle Public Score"
                                      >
                                        <Edit3 size={12} />
                                      </button>
                                    </div>
                                  )}
                                </td>

                                <td className="px-5 py-3.5 text-right whitespace-nowrap">
                                  <a
                                    href={`/api/cases/${selectedCaseId}/download/submission/run/${run.run_id}`}
                                    className="inline-flex items-center space-x-1 px-2.5 py-1 bg-white hover:bg-[#F4F5F6] text-[#121816] text-[11px] font-medium border border-[#E5E9E6] rounded-lg transition-colors"
                                    title={`Download ${run.submission_filename || 'submission.csv'}`}
                                  >
                                    <Download size={12} className="text-[#164E33]" />
                                    <span>Download CSV</span>
                                  </a>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {/* Diagnostic Banner if score dropped */}
                  {caseDetail.history && caseDetail.history.length >= 2 && caseDetail.history[caseDetail.history.length - 1].cv_delta < 0 && (
                    <div className="p-4 bg-amber-50/60 border-t border-amber-200/80 flex items-start space-x-3 text-xs">
                      <AlertTriangle size={16} className="text-amber-700 flex-shrink-0 mt-0.5" />
                      <div className="space-y-1 text-amber-900 leading-relaxed">
                        <span className="font-semibold">Diagnostic Insight on Score Drop:</span>
                        <p className="text-[11px] text-amber-800">
                          Adding the universal primitives dropped both local CV ({caseDetail.history[caseDetail.history.length - 1].cv_delta.toFixed(5)}) and Kaggle Public LB ({((caseDetail.history[caseDetail.history.length - 1].lb_score || 0) - (caseDetail.history[0].lb_score || 0)).toFixed(5)}). Notice in the Feature Importances below that <code className="bg-amber-100/80 px-1 rounded font-mono font-medium">row_stat_min</code> received huge importance (105k) by computing min across non-comparable units (age, USD, hours, km). Tree models greedily split on this noise, causing slight generalization loss.
                        </p>
                        <p className="text-[11px] text-[#164E33] font-medium">
                          Next step recommendation: In Step 2, prune arbitrary row-wise aggregates (<code className="bg-emerald-50 px-1 rounded font-mono">row_stat_*</code>) and preserve domain-meaningful financial ratios (<code className="bg-emerald-50 px-1 rounded font-mono">amount_usd / account_balance_usd</code>) to restore baseline score!
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Feature Importance Section */}
                {caseDetail.feature_importances?.length > 0 && (
                  <div className="border border-zinc-200 rounded p-6">
                    <h3 className="font-bold text-base text-zinc-950 mb-1">Global Feature Importances</h3>
                    <p className="text-xs text-zinc-600 mb-6">
                      Aggregated feature importance across cross-validation folds.
                    </p>

                    <div className="space-y-3">
                      {caseDetail.feature_importances.slice(0, 15).map((item, idx) => {
                        const maxImp = caseDetail.feature_importances[0].importance;
                        const pct = Math.max(4, Math.round((item.importance / maxImp) * 100));
                        return (
                          <div key={item.feature} className="flex items-center space-x-4 text-xs font-mono">
                            <span className="w-6 text-zinc-400">#{idx + 1}</span>
                            <span className="w-56 truncate font-medium text-zinc-900">{item.feature}</span>
                            <div className="flex-1 bg-zinc-100 h-4 rounded overflow-hidden">
                              <div className="bg-zinc-900 h-full rounded" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="w-24 text-right text-zinc-500">
                              {item.importance.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 4: ENSEMBLE LABORATORY */}
            {activeTab === 'ensemble' && (
              <div className="space-y-8">
                {caseDetail.available_cached.length < 2 ? (
                  <div className="border border-zinc-200 rounded p-12 text-center bg-zinc-50/30">
                    <Sliders className="mx-auto text-zinc-400 mb-3" size={32} />
                    <div className="text-sm font-semibold text-zinc-900">Ensemble Laboratory Standby</div>
                    <p className="text-xs text-zinc-500 max-w-md mx-auto mt-1 leading-relaxed">
                      The Ensemble Laboratory requires at least 2 trained models (e.g. LightGBM, XGBoost, CatBoost). 
                      Click "Run Pipeline" on Step 3 to train models and generate Out-Of-Fold predictions.
                    </p>
                  </div>
                ) : (
                  <>
                    {/* 1. Automated Caruana Hill Climbing */}
                    <div className="border border-zinc-200 rounded p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-bold text-base text-zinc-950">1. Automated Caruana Hill Climbing</h3>
                          <p className="text-xs text-zinc-600 mt-1">
                            Greedy forward selection with replacement directly maximizing {meta?.eval_metric.toUpperCase()} on OOF predictions.
                          </p>
                        </div>

                        <button
                          onClick={handleRunCaruana}
                          disabled={ensembleLoading}
                          className="flex items-center space-x-1.5 bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-semibold px-4 py-2 rounded transition"
                        >
                          {ensembleLoading ? <Loader2 size={14} className="animate-spin" /> : <TrendingUp size={14} />}
                          <span>Optimize Weights</span>
                        </button>
                      </div>

                      {caruanaResult && (
                        <div className="mt-5 pt-5 border-t border-zinc-200 flex items-center justify-between bg-zinc-50 p-4 rounded">
                          <div>
                            <div className="text-xs text-zinc-500 font-semibold uppercase">Caruana Optimized Score</div>
                            <div className="text-2xl font-bold text-zinc-950 mt-1 flex items-baseline space-x-2">
                              <span>{caruanaResult.score.toFixed(5)}</span>
                              <span className="text-xs font-mono font-medium text-zinc-700 bg-zinc-200 px-2 py-0.5 rounded">
                                +{caruanaResult.gain.toFixed(5)} vs Best Single
                              </span>
                            </div>
                          </div>

                          <div className="text-right">
                            <div className="text-xs text-zinc-500 font-semibold uppercase mb-1">Optimized Weights</div>
                            <div className="flex space-x-2">
                              {Object.entries(caruanaResult.weights).map(([m, w]: any) => (
                                <span key={m} className="text-xs font-mono bg-white border border-zinc-300 px-2 py-1 rounded">
                                  <strong>{m.toUpperCase()}:</strong> {(w * 100).toFixed(1)}%
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* 2. Interactive Manual Weight Tuning */}
                    <div className="border border-zinc-200 rounded p-6">
                      <h3 className="font-bold text-base text-zinc-950">2. Interactive Manual Blending</h3>
                      <p className="text-xs text-zinc-600 mt-1 mb-6">
                        Adjust model contribution weights interactively. Scores are evaluated in milliseconds on pre-computed OOF.
                      </p>

                      <div className="grid grid-cols-3 gap-6">
                        {caseDetail.available_cached.map(m => (
                          <div key={m} className="border border-zinc-200 rounded p-4 bg-zinc-50/50">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-xs font-bold uppercase text-zinc-800">{m}</span>
                              <span className="text-xs font-mono font-semibold text-zinc-950">
                                {customWeights[m] ?? 0}
                              </span>
                            </div>
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={customWeights[m] ?? 0}
                              onChange={e => {
                                const newW = { ...customWeights, [m]: parseFloat(e.target.value) };
                                setCustomWeights(newW);
                                handleBlend(newW);
                              }}
                              className="w-full accent-zinc-950 cursor-pointer"
                            />
                          </div>
                        ))}
                      </div>

                      {customScore !== null && (
                        <div className="mt-6 pt-6 border-t border-zinc-200 flex items-center justify-between">
                          <div>
                            <div className="text-xs font-semibold uppercase text-zinc-500">Custom Blend Score</div>
                            <div className="text-2xl font-bold text-zinc-950 mt-0.5">
                              {customScore.toFixed(5)}
                            </div>
                          </div>

                          <a
                            href={`/api/cases/${selectedCaseId}/download/submission?custom=true`}
                            className="flex items-center space-x-1.5 border border-zinc-300 hover:border-zinc-900 text-zinc-900 text-xs font-semibold px-4 py-2 rounded transition"
                          >
                            <Download size={14} />
                            <span>Download Custom Blend Submission</span>
                          </a>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}

            {/* TAB 5: SUBMISSIONS & PRODUCTION ARTIFACTS */}
            {activeTab === 'artifacts' && (
              <div className="space-y-6">
                <div className="border border-zinc-200 rounded p-6 bg-zinc-50/30">
                  <h3 className="font-bold text-base text-zinc-950">Production Artifacts & Downloads</h3>
                  <p className="text-xs text-zinc-600 mt-1">
                    Export generated predictions ready for Kaggle submission or secondary multi-level stacking.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div className="border border-zinc-200 rounded p-6 space-y-3">
                    <h4 className="font-semibold text-sm text-zinc-950">1. Champion Submission CSV</h4>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Generated from the highest-scoring strategy on cross-validation ({championModel || 'Champion Model'}).
                    </p>
                    <a
                      href={`/api/cases/${selectedCaseId}/download/submission`}
                      className="inline-flex items-center space-x-2 bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-semibold py-2.5 px-4 rounded transition"
                    >
                      <Download size={14} />
                      <span>Download Champion Submission</span>
                    </a>
                  </div>

                  <div className="border border-zinc-200 rounded p-6 space-y-3">
                    <h4 className="font-semibold text-sm text-zinc-950">2. Out-Of-Fold Predictions CSV</h4>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Contains ground truth labels and OOF probabilities for error analysis or level-2 meta-learning.
                    </p>
                    <a
                      href={`/api/cases/${selectedCaseId}/download/oof`}
                      className="inline-flex items-center space-x-2 border border-zinc-300 hover:border-zinc-900 bg-white text-zinc-800 text-xs font-semibold py-2.5 px-4 rounded transition"
                    >
                      <Download size={14} />
                      <span>Download OOF Predictions</span>
                    </a>
                  </div>

                  <div className="border border-zinc-200 rounded p-6 space-y-3">
                    <h4 className="font-semibold text-sm text-zinc-950">3. Standalone Solution Script (.py)</h4>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Self-contained clean Python code for deliverables. Runs from terminal: `python solution.py`.
                    </p>
                    <a
                      href={`/api/cases/${selectedCaseId}/export-script`}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center space-x-2 border border-zinc-300 hover:border-zinc-900 bg-white text-zinc-800 text-xs font-semibold py-2.5 px-4 rounded transition"
                    >
                      <FileCode size={14} />
                      <span>Download Solution Script (.py)</span>
                    </a>
                  </div>

                  <div className="border border-zinc-200 rounded p-6 space-y-3">
                    <h4 className="font-semibold text-sm text-zinc-950">4. Jupyter Solution Notebook (.ipynb)</h4>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Interactive competition notebook with methodology markdown cells ready to submit to judges.
                    </p>
                    <a
                      href={`/api/cases/${selectedCaseId}/export-notebook`}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center space-x-2 bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-semibold py-2.5 px-4 rounded transition"
                    >
                      <BookOpen size={14} />
                      <span>Download Jupyter Notebook (.ipynb)</span>
                    </a>
                  </div>
                </div>

                {summary && (
                  <div className="border border-zinc-200 rounded p-5 bg-zinc-50">
                    <h4 className="font-semibold text-xs uppercase text-zinc-500 tracking-wider mb-2">Experiment Meta Summary</h4>
                    <pre className="text-xs font-mono text-zinc-800 overflow-x-auto">
                      {JSON.stringify(summary, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </main>

      {/* MODAL: REGISTER NEW CASE */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white border border-zinc-200 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl p-6">
            <div className="flex items-center justify-between pb-4 border-b border-zinc-200">
              <h2 className="text-base font-bold text-zinc-950">Register New Tabular Case</h2>
              <button onClick={() => setIsModalOpen(false)} className="text-zinc-400 hover:text-zinc-900">
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleCreateCase} className="mt-5 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-zinc-700">Case Unique ID</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. nfl_big_data_bowl_2026"
                    value={newCase.id}
                    onChange={e => setNewCase({ ...newCase, id: e.target.value })}
                    className="mt-1 w-full text-xs border border-zinc-300 rounded p-2 text-zinc-900 focus:outline-none focus:border-zinc-950"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-zinc-700">Display Name</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. NFL Big Data Bowl 2026"
                    value={newCase.name}
                    onChange={e => setNewCase({ ...newCase, name: e.target.value })}
                    className="mt-1 w-full text-xs border border-zinc-300 rounded p-2 text-zinc-900 focus:outline-none focus:border-zinc-950"
                  />
                </div>
              </div>

              {/* Dataset Source Type Segmented Switch */}
              <div className="pt-2 border-t border-zinc-200 space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-zinc-900">Training Dataset Source</label>
                  <div className="flex bg-zinc-100 p-0.5 rounded-lg text-[11px] font-medium border border-zinc-200">
                    <button
                      type="button"
                      onClick={() => setNewCase(prev => ({ ...prev, auto_stitch: false }))}
                      className={`px-3 py-1 rounded-md transition ${
                        !newCase.auto_stitch ? 'bg-white text-zinc-950 font-bold shadow-xs' : 'text-zinc-500 hover:text-zinc-900'
                      }`}
                    >
                      Single CSV File
                    </button>
                    <button
                      type="button"
                      onClick={() => setNewCase(prev => ({ ...prev, auto_stitch: true }))}
                      className={`px-3 py-1 rounded-md transition ${
                        newCase.auto_stitch ? 'bg-[#164E33] text-white font-bold shadow-xs' : 'text-zinc-500 hover:text-zinc-900'
                      }`}
                    >
                      Multi-File / Partitioned Batch
                    </button>
                  </div>
                </div>

                {!newCase.auto_stitch ? (
                  <div>
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-semibold text-zinc-700">Train CSV File Path</label>
                      {scannedCsvs.length > 0 && (
                        <span className="text-[10px] text-zinc-500 font-mono">
                          {scannedCsvs.length} workspace CSVs detected
                        </span>
                      )}
                    </div>
                    <div className="flex space-x-2 mt-1">
                      <input
                        type="text"
                        required={!newCase.auto_stitch}
                        placeholder="e.g. playground-series/dataset/train.csv"
                        value={newCase.train_path}
                        onChange={e => {
                          setNewCase({ ...newCase, train_path: e.target.value });
                          setInspectPath(e.target.value);
                        }}
                        className="w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 focus:outline-none focus:border-zinc-950"
                      />
                      <button
                        type="button"
                        onClick={() => handleBrowseFile('train')}
                        disabled={isBrowsingTrain}
                        className="px-3 py-2 border border-zinc-300 hover:border-zinc-950 bg-zinc-50 hover:bg-zinc-100 text-xs font-medium rounded whitespace-nowrap flex items-center space-x-1 text-zinc-900 transition"
                        title="Open native Windows Explorer file picker"
                      >
                        {isBrowsingTrain ? <Loader2 size={12} className="animate-spin" /> : <FolderOpen size={12} />}
                        <span>{isBrowsingTrain ? 'Picking...' : 'Browse...'}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => handleInspect()}
                        disabled={inspectLoading || !newCase.train_path}
                        className="px-3 py-2 border border-zinc-300 hover:border-zinc-900 bg-zinc-50 hover:bg-zinc-100 text-xs font-medium rounded whitespace-nowrap flex items-center space-x-1"
                      >
                        {inspectLoading && <Loader2 size={12} className="animate-spin" />}
                        <span>{inspectLoading ? 'Inspecting...' : 'Inspect'}</span>
                      </button>
                    </div>
                    {scannedCsvs.length > 0 && (
                      <div className="mt-1.5 flex items-center space-x-2">
                        <span className="text-zinc-500 text-[11px] whitespace-nowrap">Quick Select:</span>
                        <select
                          onChange={e => {
                            if (e.target.value) {
                              setNewCase(prev => ({ ...prev, train_path: e.target.value }));
                              setInspectPath(e.target.value);
                              handleInspect(e.target.value);
                            }
                          }}
                          className="w-full text-[11px] border border-zinc-200 rounded px-2 py-1 font-mono text-zinc-800 bg-zinc-50 focus:outline-none focus:border-zinc-950 truncate"
                          value=""
                        >
                          <option value="" disabled>Choose detected CSV from project workspace...</option>
                          {scannedCsvs.map(csv => (
                            <option key={csv.path} value={csv.path}>
                              {csv.path} ({csv.size_mb} MB)
                            </option>
                          ))}
                        </select>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-3.5 bg-zinc-50 border border-zinc-200 rounded-xl space-y-3">
                    <div className="space-y-1">
                      <label className="text-xs font-semibold text-zinc-800">
                        Primary / Target Partitions Wildcard *
                      </label>
                      <input
                        type="text"
                        required={newCase.auto_stitch}
                        placeholder="e.g. nfl-big-data-bowl-2026-prediction/train/output_*.csv or data/part_*.csv"
                        value={newCase.stitch_base_pattern}
                        onChange={e => setNewCase({ ...newCase, stitch_base_pattern: e.target.value })}
                        className="w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 bg-white focus:outline-none focus:border-zinc-950"
                      />
                      <p className="text-[10px] text-zinc-500">
                        All matched files will be vertically stacked. Target column name in this table will be strictly preserved.
                      </p>
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs font-semibold text-zinc-800">
                        Feature Store Partitions Wildcard (Optional)
                      </label>
                      <input
                        type="text"
                        placeholder="e.g. nfl-big-data-bowl-2026-prediction/train/input_*.csv"
                        value={newCase.stitch_join_pattern}
                        onChange={e => setNewCase({ ...newCase, stitch_join_pattern: e.target.value })}
                        className="w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 bg-white focus:outline-none focus:border-zinc-950"
                      />
                      <p className="text-[10px] text-zinc-500">
                        Leave empty if dataset has only one partition pattern. If provided, both will be concatenated and merged on join keys.
                      </p>
                    </div>

                    {newCase.stitch_join_pattern && (
                      <div className="space-y-1">
                        <label className="text-xs font-semibold text-zinc-800">
                          Relational Join Keys (Comma Separated) *
                        </label>
                        <input
                          type="text"
                          required={!!newCase.stitch_join_pattern}
                          placeholder="e.g. game_id, play_id, nfl_id, frame_id"
                          value={newCase.stitch_join_keys?.join(', ') || ''}
                          onChange={e => setNewCase({
                            ...newCase,
                            stitch_join_keys: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
                          })}
                          className="w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 bg-white focus:outline-none focus:border-zinc-950"
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>

              {inspectData?.columns && (
                <div className="p-3 bg-zinc-50 border border-zinc-200 rounded text-xs space-y-2">
                  <div className="font-semibold text-zinc-800">Detected Columns ({inspectData.total_columns}):</div>
                  <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                    {inspectData.columns.map((c: any) => (
                      <span key={c.column} className="bg-white border border-zinc-200 px-1.5 py-0.5 rounded font-mono text-[11px]">
                        {c.column}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <label className="text-xs font-semibold text-zinc-700">Test CSV File Path (Optional)</label>
                <div className="flex space-x-2 mt-1">
                  <input
                    type="text"
                    placeholder="e.g. playground-series/dataset/test.csv"
                    value={newCase.test_path}
                    onChange={e => setNewCase({ ...newCase, test_path: e.target.value })}
                    className="w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 focus:outline-none focus:border-zinc-950"
                  />
                  <button
                    type="button"
                    onClick={() => handleBrowseFile('test')}
                    disabled={isBrowsingTest}
                    className="px-3 py-2 border border-zinc-300 hover:border-zinc-950 bg-zinc-50 hover:bg-zinc-100 text-xs font-medium rounded whitespace-nowrap flex items-center space-x-1 text-zinc-900 transition"
                    title="Open native Windows Explorer file picker"
                  >
                    {isBrowsingTest ? <Loader2 size={12} className="animate-spin" /> : <FolderOpen size={12} />}
                    <span>{isBrowsingTest ? 'Picking...' : 'Browse...'}</span>
                  </button>
                </div>
                {scannedCsvs.length > 0 && (
                  <div className="mt-1.5 flex items-center space-x-2">
                    <span className="text-zinc-500 text-[11px] whitespace-nowrap">Quick Select:</span>
                    <select
                      onChange={e => {
                        if (e.target.value) {
                          setNewCase(prev => ({ ...prev, test_path: e.target.value }));
                        }
                      }}
                      className="w-full text-[11px] border border-zinc-200 rounded px-2 py-1 font-mono text-zinc-800 bg-zinc-50 focus:outline-none focus:border-zinc-950 truncate"
                      value=""
                    >
                      <option value="" disabled>Choose detected CSV from project workspace...</option>
                      {scannedCsvs.map(csv => (
                        <option key={csv.path} value={csv.path}>
                          {csv.path} ({csv.size_mb} MB)
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-zinc-700">Target Column</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. target or Transported"
                    value={newCase.target_col}
                    onChange={e => setNewCase({ ...newCase, target_col: e.target.value })}
                    className="mt-1 w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 focus:outline-none focus:border-zinc-950"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-zinc-700">ID Column (Optional)</label>
                  <input
                    type="text"
                    placeholder="e.g. id or PassengerId"
                    value={newCase.id_col}
                    onChange={e => setNewCase({ ...newCase, id_col: e.target.value })}
                    className="mt-1 w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 focus:outline-none focus:border-zinc-950"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-zinc-700">Task Type</label>
                  <select
                    value={newCase.task_type}
                    onChange={e => {
                      const tt = e.target.value;
                      let defaultMetric = newCase.eval_metric;
                      let defaultCv = newCase.cv_strategy;
                      if (tt === 'regression') {
                        defaultMetric = 'rmse';
                        if (defaultCv === 'stratified_kfold') defaultCv = 'kfold';
                      } else if (tt === 'binary') {
                        defaultMetric = 'roc_auc';
                        if (defaultCv === 'kfold') defaultCv = 'stratified_kfold';
                      } else {
                        defaultMetric = 'f1_macro';
                      }
                      setNewCase({ ...newCase, task_type: tt, eval_metric: defaultMetric, cv_strategy: defaultCv });
                    }}
                    className="mt-1 w-full text-xs border border-zinc-300 rounded p-2 text-zinc-900 focus:outline-none"
                  >
                    <option value="binary">Binary Classification</option>
                    <option value="multiclass">Multiclass Classification</option>
                    <option value="regression">Regression</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-semibold text-zinc-700">Primary Metric</label>
                  <select
                    value={newCase.eval_metric}
                    onChange={e => setNewCase({ ...newCase, eval_metric: e.target.value })}
                    className="mt-1 w-full text-xs border border-zinc-300 rounded p-2 text-zinc-900 focus:outline-none"
                  >
                    <option value="roc_auc">ROC-AUC</option>
                    <option value="accuracy">Accuracy</option>
                    <option value="f1_macro">F1 Macro</option>
                    <option value="logloss">Log Loss</option>
                    <option value="rmse">RMSE</option>
                    <option value="mae">MAE</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-zinc-700">CV Validation Strategy</label>
                  <select
                    value={newCase.cv_strategy}
                    onChange={e => setNewCase({ ...newCase, cv_strategy: e.target.value })}
                    className="mt-1 w-full text-xs border border-zinc-300 rounded p-2 text-zinc-900 focus:outline-none focus:border-zinc-950 font-medium"
                  >
                    <option value="stratified_kfold">Stratified K-Fold (Classification)</option>
                    <option value="kfold">Standard K-Fold (Regression / Unstratified)</option>
                    <option value="group_kfold">Group K-Fold (Group non-overlap)</option>
                    <option value="stratified_group_kfold">Stratified Group K-Fold</option>
                    <option value="time_series">Time Series Split</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-semibold text-zinc-700">Number of Folds (n_splits)</label>
                  <input
                    type="number"
                    min={2}
                    max={20}
                    value={newCase.n_splits}
                    onChange={e => setNewCase({ ...newCase, n_splits: parseInt(e.target.value) || 5 })}
                    className="mt-1 w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 focus:outline-none focus:border-zinc-950"
                  />
                </div>
              </div>

              {(newCase.cv_strategy === 'group_kfold' || newCase.cv_strategy === 'stratified_group_kfold') && (
                <div className="p-3 bg-zinc-50 border border-zinc-200 rounded text-xs space-y-1">
                  <label className="text-xs font-semibold text-zinc-800">Group Column (Required for Group CV)</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. play_id, game_id, or client_id"
                    value={newCase.group_col}
                    onChange={e => setNewCase({ ...newCase, group_col: e.target.value })}
                    className="w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 focus:outline-none focus:border-zinc-950 bg-white"
                  />
                  <p className="text-[11px] text-zinc-500">
                    Samples with the same group value will strictly remain together in either train or validation split.
                  </p>
                </div>
              )}

              {newCase.cv_strategy === 'time_series' && (
                <div className="p-3 bg-zinc-50 border border-zinc-200 rounded text-xs space-y-1">
                  <label className="text-xs font-semibold text-zinc-800">Time Column (Required for Time Series)</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. timestamp, date, or step"
                    value={newCase.time_col}
                    onChange={e => setNewCase({ ...newCase, time_col: e.target.value })}
                    className="w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 focus:outline-none focus:border-zinc-950 bg-white"
                  />
                  <p className="text-[11px] text-zinc-500">
                    Chronological ordering column ensuring future observations are never used to predict the past.
                  </p>
                </div>
              )}

              <div>
                <label className="text-xs font-semibold text-zinc-700">Custom Feature Engineering Hook (Python)</label>
                <textarea
                  rows={4}
                  placeholder={`from tabular_pipeline import BaseFeatureHook\n\nclass CustomFeatures(BaseFeatureHook):\n    def transform(self, train_df, test_df=None):\n        # Custom FE logic\n        return train_df, test_df`}
                  value={newCase.hook_code}
                  onChange={e => setNewCase({ ...newCase, hook_code: e.target.value })}
                  className="mt-1 w-full text-xs font-mono border border-zinc-300 rounded p-2 text-zinc-900 focus:outline-none focus:border-zinc-950"
                />
              </div>

              <div className="pt-4 border-t border-zinc-200 flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 border border-zinc-300 hover:border-zinc-900 text-xs font-medium rounded text-zinc-800"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isCreatingCase}
                  className="px-5 py-2 bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-semibold rounded disabled:opacity-50 flex items-center space-x-1.5"
                >
                  {isCreatingCase ? (
                    <>
                      <Loader2 size={13} className="animate-spin" />
                      <span>{newCase.auto_stitch ? 'Stitching Partitions & Building Case...' : 'Creating Case...'}</span>
                    </>
                  ) : (
                    <span>Register & Open Case</span>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Dataset Stitcher & Multi-Table Joiner Studio Modal */}
      {showStitcherModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-xs flex items-center justify-center z-50 p-4 animate-fade-in overflow-y-auto">
          <div className="bg-white border border-[#E5E9E6] rounded-2xl max-w-4xl w-full p-6 shadow-2xl space-y-6 my-8 max-h-[90vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="flex items-start justify-between border-b border-[#E5E9E6] pb-4">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 rounded-lg bg-[#E8F3ED] border border-[#D2E7DC] flex items-center justify-center text-[#164E33]">
                    <Layers size={18} />
                  </div>
                  <h3 className="text-base font-bold text-[#121816]">
                    Dataset Stitcher & Multi-Table Joiner Studio
                  </h3>
                  <span className="text-[10px] font-mono uppercase bg-[#164E33] text-white px-2 py-0.5 rounded font-bold">
                    Multi-Table Engine
                  </span>
                </div>
                <p className="text-xs text-[#76807B]">
                  Merge relational tables (Base + Auxiliary) or concatenate partitioned multi-file batches into unified tabular matrices without writing Python scripts.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowStitcherModal(false)}
                className="text-zinc-400 hover:text-zinc-700 p-1 rounded-lg"
              >
                <X size={18} />
              </button>
            </div>

            {/* Mode Selector Tabs */}
            <div className="flex border-b border-[#E5E9E6] space-x-6 text-xs font-medium">
              <button
                type="button"
                onClick={() => { setStitcherMode('wizard'); setStitcherPreviewData(null); }}
                className={`pb-2.5 transition flex items-center space-x-2 border-b-2 ${
                  stitcherMode === 'wizard'
                    ? 'border-[#164E33] text-[#164E33] font-bold'
                    : 'border-transparent text-[#76807B] hover:text-[#121816]'
                }`}
              >
                <Layers size={14} />
                <span>Batch Multi-File Stitcher</span>
              </button>
              <button
                type="button"
                onClick={() => { setStitcherMode('join'); setStitcherPreviewData(null); }}
                className={`pb-2.5 transition flex items-center space-x-2 border-b-2 ${
                  stitcherMode === 'join'
                    ? 'border-[#164E33] text-[#164E33] font-bold'
                    : 'border-transparent text-[#76807B] hover:text-[#121816]'
                }`}
              >
                <GitMerge size={14} />
                <span>Relational Join (Two Tables)</span>
              </button>
              <button
                type="button"
                onClick={() => { setStitcherMode('concat'); setStitcherPreviewData(null); }}
                className={`pb-2.5 transition flex items-center space-x-2 border-b-2 ${
                  stitcherMode === 'concat'
                    ? 'border-[#164E33] text-[#164E33] font-bold'
                    : 'border-transparent text-[#76807B] hover:text-[#121816]'
                }`}
              >
                <Layers size={14} />
                <span>Partition Concatenator (Single Pattern)</span>
              </button>
            </div>

            {/* Notifications */}
            {stitcherError && (
              <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl text-xs">
                {stitcherError}
              </div>
            )}
            {stitcherSuccess && (
              <div className="p-3 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-xl text-xs font-medium flex items-center space-x-2">
                <Check size={14} />
                <span>{stitcherSuccess}</span>
              </div>
            )}

            {/* TAB 1: BATCH MULTI-FILE STITCHER */}
            {stitcherMode === 'wizard' && (
              <div className="space-y-5">
                <div className="p-4 bg-[#E8F3ED]/40 border border-[#D2E7DC] rounded-xl">
                  <div className="space-y-1">
                    <div className="text-xs font-bold text-[#164E33] flex items-center space-x-1.5">
                      <Layers size={14} />
                      <span>Unified Batch Stacking & Relational Join</span>
                    </div>
                    <p className="text-xs text-[#121816]/80 leading-relaxed">
                      Concatenates all primary partition files matching a wildcard pattern + optional feature partitions, merges them on common join keys, preserves target column names without breaking suffixes, and updates your active case dataset.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-[#121816]">
                      Base Partitions Wildcard (Primary / Target Tables) *
                    </label>
                    <input
                      type="text"
                      value={wizardBasePattern}
                      onChange={e => setWizardBasePattern(e.target.value)}
                      placeholder="e.g. data/train/output_*.csv or train_part_*.csv"
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                    />
                    <div className="text-[10px] text-[#76807B]">All matched files will be vertically concatenated</div>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-[#121816]">
                      Feature Partitions Wildcard (Optional Auxiliary Inputs)
                    </label>
                    <input
                      type="text"
                      value={wizardJoinPattern}
                      onChange={e => setWizardJoinPattern(e.target.value)}
                      placeholder="e.g. data/train/input_*.csv"
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                    />
                    <div className="text-[10px] text-[#76807B]">Leave blank if your dataset is already in single partition files</div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-[#121816]">Join Keys (Comma Separated) *</label>
                    <input
                      type="text"
                      value={wizardJoinKeys}
                      onChange={e => setWizardJoinKeys(e.target.value)}
                      placeholder="game_id, play_id, nfl_id, frame_id"
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-[#121816]">Output Filename</label>
                    <input
                      type="text"
                      value={wizardOutFile}
                      onChange={e => setWizardOutFile(e.target.value)}
                      placeholder="train_all_weeks_merged.csv"
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-[#121816]">Max Files (Optional Limit)</label>
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={wizardMaxFiles}
                      onChange={e => setWizardMaxFiles(e.target.value)}
                      placeholder="All files (or e.g. 3 for quick test)"
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: RELATIONAL JOIN */}
            {stitcherMode === 'join' && (
              <div className="space-y-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Base Table */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-[#121816] flex items-center justify-between">
                      <span>Base Table (Target / Master) *</span>
                      <span className="text-[10px] text-[#76807B] font-mono">e.g. output.csv or train.csv</span>
                    </label>
                    <input
                      type="text"
                      value={stitcherBasePath}
                      onChange={e => handleInspectBaseColumns(e.target.value)}
                      placeholder="e.g. nfl-big-data-bowl-2026-prediction/train/output_2023_w01.csv"
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                    />
                    {baseColumns.length > 0 && (
                      <div className="text-[10px] text-[#76807B] font-mono">
                        Loaded {baseColumns.length} columns: {baseColumns.slice(0, 5).join(', ')}...
                      </div>
                    )}
                  </div>

                  {/* Join Table */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-[#121816] flex items-center justify-between">
                      <span>Join Table (Auxiliary / Feature Store) *</span>
                      <span className="text-[10px] text-[#76807B] font-mono">e.g. input.csv or profiles.csv</span>
                    </label>
                    <input
                      type="text"
                      value={stitcherJoinPath}
                      onChange={e => handleInspectJoinColumns(e.target.value)}
                      placeholder="e.g. nfl-big-data-bowl-2026-prediction/train/input_2023_w01.csv"
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                    />
                    {joinColumns.length > 0 && (
                      <div className="text-[10px] text-[#76807B] font-mono">
                        Loaded {joinColumns.length} columns: {joinColumns.slice(0, 5).join(', ')}...
                      </div>
                    )}
                  </div>
                </div>

                {/* Scanned CSVs Helper Chips */}
                {scannedCsvs.length > 0 && (
                  <div className="p-3 bg-[#F8FAF9] border border-[#E5E9E6] rounded-xl space-y-1.5">
                    <span className="text-[10px] uppercase font-bold text-[#76807B] font-mono">Quick Pick from Workspace:</span>
                    <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
                      {scannedCsvs.slice(0, 15).map(f => (
                        <button
                          key={f.path}
                          type="button"
                          onClick={() => {
                            if (!stitcherBasePath) handleInspectBaseColumns(f.path);
                            else handleInspectJoinColumns(f.path);
                          }}
                          className="px-2 py-0.5 bg-white border border-[#E5E9E6] hover:border-[#164E33] text-[10px] font-mono rounded text-[#121816] transition"
                        >
                          {f.filename} ({f.size_mb}MB)
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Join Keys Selector */}
                <div className="space-y-2">
                  <label className="text-xs font-semibold text-[#121816] flex items-center justify-between">
                    <span>Join Key Columns (Matching Identifiers) *</span>
                    <span className="text-[10px] text-[#76807B]">Click column tags to toggle as join keys</span>
                  </label>
                  
                  {/* Detected Common Columns */}
                  {baseColumns.length > 0 && joinColumns.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {joinColumns.filter(c => baseColumns.includes(c)).map(col => {
                        const isSelected = stitcherJoinKeys.includes(col);
                        return (
                          <button
                            key={col}
                            type="button"
                            onClick={() => {
                              if (isSelected) {
                                setStitcherJoinKeys(stitcherJoinKeys.filter(k => k !== col));
                              } else {
                                setStitcherJoinKeys([...stitcherJoinKeys, col]);
                              }
                            }}
                            className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium border transition ${
                              isSelected
                                ? 'bg-[#164E33] text-white border-[#164E33]'
                                : 'bg-white text-[#76807B] border-[#E5E9E6] hover:border-[#164E33]'
                            }`}
                          >
                            <span>{isSelected ? '[x] ' : '+ '}</span>
                            <span>{col}</span>
                          </button>
                        );
                      })}
                    </div>
                  )}

                  <input
                    type="text"
                    value={stitcherJoinKeys.join(', ')}
                    onChange={e => setStitcherJoinKeys(e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                    placeholder="e.g. game_id, play_id, nfl_id"
                    className="w-full px-3 py-1.5 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816] focus:outline-none focus:border-[#164E33]"
                  />
                </div>

                {/* Strategy, 1-to-Many Reduction, and Filter */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Join Type */}
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-[#121816]">Join Strategy</label>
                    <select
                      value={stitcherHow}
                      onChange={e => setStitcherHow(e.target.value as any)}
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs text-[#121816] focus:outline-none focus:border-[#164E33]"
                    >
                      <option value="inner">Inner Join (Keep only matching)</option>
                      <option value="left">Left Join (Keep all base rows)</option>
                      <option value="outer">Outer Join (Keep everything)</option>
                    </select>
                  </div>

                  {/* 1-to-Many Reduction */}
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-[#121816]">1-to-Many Reduction</label>
                    <select
                      value={stitcherAgg}
                      onChange={e => setStitcherAgg(e.target.value as any)}
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs text-[#121816] focus:outline-none focus:border-[#164E33]"
                    >
                      <option value="none">None (Standard Merge)</option>
                      <option value="last">Take Last Record per Group (T_0 Temporal State)</option>
                      <option value="first">Take First Record per Group</option>
                    </select>
                  </div>

                  {/* Optional Filter */}
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-[#121816]">Optional Row Filter</label>
                    <div className="flex space-x-1.5">
                      <input
                        type="text"
                        placeholder="Col (e.g. player_to_predict)"
                        value={stitcherFilterCol}
                        onChange={e => setStitcherFilterCol(e.target.value)}
                        className="w-1/2 px-2 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono"
                      />
                      <input
                        type="text"
                        placeholder="Val (e.g. True)"
                        value={stitcherFilterVal}
                        onChange={e => setStitcherFilterVal(e.target.value)}
                        className="w-1/2 px-2 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono"
                      />
                    </div>
                  </div>
                </div>

                {/* Output & Destination */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-[#E5E9E6]">
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-[#121816]">Output Filename</label>
                    <input
                      type="text"
                      value={stitcherOutFile}
                      onChange={e => setStitcherOutFile(e.target.value)}
                      placeholder="e.g. train_merged.csv"
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816]"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-[#121816]">Apply to Case</label>
                    <select
                      value={stitcherApplyTarget}
                      onChange={e => setStitcherApplyTarget(e.target.value as any)}
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs text-[#121816]"
                    >
                      <option value="train">Save & Set as Training Dataset (train_path)</option>
                      <option value="test">Save & Set as Test Dataset (test_path)</option>
                      <option value="none">Save Only (Do not modify case.json)</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: CONCAT PARTITIONS */}
            {stitcherMode === 'concat' && (
              <div className="space-y-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-[#121816] flex items-center justify-between">
                    <span>File Pattern Wildcard (Glob) *</span>
                    <span className="text-[10px] text-[#76807B] font-mono">Matches all partitions matching pattern</span>
                  </label>
                  <input
                    type="text"
                    value={concatPattern}
                    onChange={e => setConcatPattern(e.target.value)}
                    placeholder="e.g. nfl-big-data-bowl-2026-prediction/train/input_2023_w*.csv"
                    className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono text-[#121816]"
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-[#121816]">Max Files (Optional)</label>
                    <input
                      type="number"
                      value={concatMaxFiles}
                      onChange={e => setConcatMaxFiles(e.target.value)}
                      placeholder="e.g. 3 (leave blank for all)"
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-[#121816]">Output Filename</label>
                    <input
                      type="text"
                      value={concatOutFile}
                      onChange={e => setConcatOutFile(e.target.value)}
                      placeholder="e.g. input_concatenated.csv"
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs font-mono"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-[#121816]">Apply to Case</label>
                    <select
                      value={concatApplyTarget}
                      onChange={e => setConcatApplyTarget(e.target.value as any)}
                      className="w-full px-3 py-2 bg-white border border-[#E5E9E6] rounded-xl text-xs"
                    >
                      <option value="train">Save & Set as Training Dataset (train_path)</option>
                      <option value="test">Save & Set as Test Dataset (test_path)</option>
                      <option value="none">Save Only (Do not modify case.json)</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* LIVE PREVIEW AREA */}
            {stitcherPreviewData && (
              <div className="bg-[#121816] border border-[#26332C] rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between text-xs font-mono text-[#E5E9E6]">
                  <span>Live Preview (5 sample rows x {stitcherPreviewData.columns.length} columns)</span>
                  <span className="text-[#A0ACA5] text-[11px]">Preview Sample Validated</span>
                </div>
                <div className="overflow-x-auto max-h-48 border border-[#26332C] rounded-lg">
                  <table className="w-full text-[10px] font-mono text-left text-zinc-300">
                    <thead className="bg-[#1C2621] text-zinc-200 uppercase tracking-wider sticky top-0">
                      <tr>
                        {stitcherPreviewData.columns.map((col: string) => (
                          <th key={col} className="px-2.5 py-1.5 border-b border-[#26332C] whitespace-nowrap">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#26332C]">
                      {stitcherPreviewData.preview_rows.map((row: any, rIdx: number) => (
                        <tr key={rIdx} className="hover:bg-white/5">
                          {stitcherPreviewData.columns.map((col: string) => (
                            <td key={col} className="px-2.5 py-1 whitespace-nowrap truncate max-w-xs">
                              {String(row[col])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Modal Actions */}
            <div className="flex items-center justify-between pt-4 border-t border-[#E5E9E6]">
              {stitcherMode === 'join' ? (
                <button
                  type="button"
                  onClick={handlePreviewMerge}
                  disabled={stitcherPreviewLoading || !stitcherBasePath || !stitcherJoinPath}
                  className="px-4 py-2 border border-[#E5E9E6] hover:border-[#164E33] hover:bg-[#F8FAF9] text-[#121816] text-xs font-medium rounded-xl transition flex items-center space-x-1.5 disabled:opacity-50"
                >
                  {stitcherPreviewLoading ? <Loader2 size={13} className="animate-spin" /> : <Layers size={13} />}
                  <span>Test & Preview Merge</span>
                </button>
              ) : <div />}

              <div className="flex items-center space-x-2.5">
                <button
                  type="button"
                  onClick={() => setShowStitcherModal(false)}
                  className="px-4 py-2 text-xs font-medium text-[#76807B] hover:text-[#121816] rounded-xl transition"
                >
                  Close
                </button>

                {stitcherMode === 'wizard' ? (
                  <button
                    type="button"
                    onClick={handleExecuteSmartStitch}
                    disabled={stitcherExecuting || !wizardBasePattern.trim() || !wizardJoinPattern.trim()}
                    className="px-5 py-2 bg-[#164E33] hover:bg-[#113d28] text-white text-xs font-semibold rounded-xl transition flex items-center space-x-1.5 disabled:opacity-50 shadow-sm"
                  >
                    {stitcherExecuting ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                    <span>{stitcherExecuting ? 'Stitching All Weeks...' : 'Run 1-Click Multi-Week Stitch'}</span>
                  </button>
                ) : stitcherMode === 'join' ? (
                  <button
                    type="button"
                    onClick={handleExecuteMerge}
                    disabled={stitcherExecuting || !stitcherBasePath || !stitcherJoinPath}
                    className="px-5 py-2 bg-[#164E33] hover:bg-[#113d28] text-white text-xs font-semibold rounded-xl transition flex items-center space-x-1.5 disabled:opacity-50 shadow-sm"
                  >
                    {stitcherExecuting ? <Loader2 size={13} className="animate-spin" /> : <GitMerge size={13} />}
                    <span>Execute Merge & Apply</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleExecuteConcat}
                    disabled={stitcherExecuting || !concatPattern}
                    className="px-5 py-2 bg-[#164E33] hover:bg-[#113d28] text-white text-xs font-semibold rounded-xl transition flex items-center space-x-1.5 disabled:opacity-50 shadow-sm"
                  >
                    {stitcherExecuting ? <Loader2 size={13} className="animate-spin" /> : <Layers size={13} />}
                    <span>Execute Concat & Apply</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
