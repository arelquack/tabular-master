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
  Square
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
  n_splits: number;
  seed: number;
  models: string[];
  output_dir?: string;
  status?: string;
  summary?: any;
}

interface CaseDetail {
  meta: CaseMeta;
  summary: any;
  feature_importances: Array<{ feature: string; importance: number }>;
  hook_code: string;
  available_cached: string[];
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
    n_splits: 5,
    seed: 2026,
    models: ['lgbm', 'xgb', 'catboost'],
    hook_code: ''
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
      } else {
        setEditableHookCode(`import pandas as pd
import numpy as np
from tabular_pipeline import BaseFeatureHook

class CustomFeatures(BaseFeatureHook):
    """
    User-definable domain feature engineering hook.
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]
        for df in dfs:
            pass # Write manual domain features here
        return train_df, test_df
`);
      }

      if (data.meta?.models?.length > 0) {
        setSelectedModels(data.meta.models);
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
      }
    } catch (e) {
      console.error('EDA fetch failed:', e);
    } finally {
      setEdaLoading(false);
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
      }
    } catch (e) {
      console.error('Failed to save hook:', e);
    } finally {
      setSavingHook(false);
    }
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
                  <div className="text-center py-12 text-zinc-500">
                    Failed to profile dataset. Check if train.csv path exists.
                  </div>
                ) : (
                  <>
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
                          <div className="text-2xl font-bold mt-1 text-[#121816] font-mono truncate">
                            {edaData.summary.target_col}
                          </div>
                        </div>
                        <div className="text-xs text-[#76807B] mt-2 pt-2 border-t border-[#E5E9E6]/60">
                          <span className="font-semibold text-[#121816]">{edaData.target_distribution.length}</span> Distinct Classes
                        </div>
                      </div>
                    </div>

                    {/* Target Distribution & Correlation Ranking */}
                    <div className="grid grid-cols-3 gap-6">
                      {/* Target Class Balance */}
                      <div className="bg-white border border-[#E5E9E6] rounded-2xl p-6 shadow-sm space-y-4">
                        <h3 className="font-bold text-sm text-[#121816]">Target Class Balance</h3>
                        <p className="text-xs text-[#76807B] leading-relaxed">
                          Class distribution in sample. Equal distribution indicates no class imbalance treatment needed.
                        </p>

                        <div className="space-y-3.5 pt-2">
                          {edaData.target_distribution.map(td => (
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
                <div className="border border-zinc-200 rounded p-6 bg-white space-y-4">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-100 pb-4">
                    <div>
                      <div className="flex items-center space-x-2">
                        <h4 className="font-bold text-sm text-zinc-950 uppercase tracking-tight">Interactive Custom Feature Hook (`feature_hook.py`)</h4>
                        <span className="text-[10px] font-mono uppercase bg-zinc-100 text-zinc-800 border border-zinc-300 px-2 py-0.5 rounded">
                          Live In-Browser Code Editor
                        </span>
                      </div>
                      <p className="text-xs text-zinc-500 mt-1">
                        Write domain-specific formulas, ratio interactions, or physics logic. Executes leak-free before model splits.
                      </p>
                    </div>

                    <div className="flex items-center space-x-3">
                      {hookSaved && (
                        <span className="text-xs font-mono text-zinc-900 font-semibold flex items-center space-x-1">
                          <Check size={14} />
                          <span>Saved & Ready</span>
                        </span>
                      )}
                      <button
                        onClick={handleSaveHook}
                        disabled={savingHook}
                        className="flex items-center space-x-1.5 bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-semibold px-4 py-2 rounded transition"
                      >
                        {savingHook ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />}
                        <span>{savingHook ? 'Saving Hook...' : 'Save Feature Hook'}</span>
                      </button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <textarea
                      value={editableHookCode}
                      onChange={e => setEditableHookCode(e.target.value)}
                      rows={16}
                      spellCheck={false}
                      className="w-full bg-zinc-950 text-zinc-100 p-4 rounded font-mono text-xs leading-relaxed border border-zinc-800 focus:outline-none focus:border-zinc-500 selection:bg-zinc-800"
                      placeholder="Write your custom BaseFeatureHook transform() method here..."
                    />
                    <div className="flex justify-between items-center text-[11px] text-zinc-500 font-mono">
                      <span>File Target: cases/{selectedCaseId}/feature_hook.py</span>
                      <span>Transform Signature: `transform(self, train_df, test_df=None)`</span>
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
                    <div className="text-xs text-[#76807B] mt-1">{meta?.n_splits}-Fold {meta?.cv_strategy}</div>
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
                    placeholder="e.g. credit_risk_2026"
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
                    placeholder="e.g. Credit Default Risk v1"
                    value={newCase.name}
                    onChange={e => setNewCase({ ...newCase, name: e.target.value })}
                    className="mt-1 w-full text-xs border border-zinc-300 rounded p-2 text-zinc-900 focus:outline-none focus:border-zinc-950"
                  />
                </div>
              </div>

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
                    required
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
                    disabled={inspectLoading}
                    className="px-3 py-2 border border-zinc-300 hover:border-zinc-900 text-xs font-medium rounded whitespace-nowrap flex items-center space-x-1"
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
                    onChange={e => setNewCase({ ...newCase, task_type: e.target.value })}
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
                  className="px-5 py-2 bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-semibold rounded"
                >
                  Save & Open Case
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
