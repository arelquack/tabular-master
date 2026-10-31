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
  Check
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

  // Optuna Tuning state
  const [tuneModel, setTuneModel] = useState<'lgbm' | 'xgb' | 'catboost'>('lgbm');
  const [tuneTrials, setTuneTrials] = useState<number>(15);
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

  // Execute Pipeline
  const handleExecute = async () => {
    if (!selectedCaseId) return;
    setRunning(true);
    setActiveTab('training');
    try {
      await fetch(`/api/cases/${selectedCaseId}/run`, { method: 'POST' });
    } catch (e) {
      console.error(e);
      setRunning(false);
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

  // Inspect Dataset file (in modal)
  const handleInspect = async () => {
    if (!inspectPath) return;
    setInspectLoading(true);
    try {
      const res = await fetch('/api/inspect-dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: inspectPath })
      });
      const data = await res.json();
      setInspectData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setInspectLoading(false);
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
    <div className="min-h-screen bg-white text-zinc-950 font-sans antialiased">
      {/* Top Header */}
      <header className="border-b border-zinc-200 bg-white sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div>
              <span className="font-bold tracking-tight text-lg text-zinc-950">TABULAR STUDIO</span>
            </div>

            {/* Case Selector */}
            <div className="flex items-center space-x-2">
              <select
                value={selectedCaseId}
                onChange={e => setSelectedCaseId(e.target.value)}
                className="bg-zinc-50 border border-zinc-300 rounded px-3 py-1.5 text-sm font-medium text-zinc-900 focus:outline-none focus:border-zinc-900"
              >
                {cases.map(c => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
              
              <button
                onClick={() => setIsModalOpen(true)}
                className="flex items-center space-x-1 border border-zinc-300 hover:border-zinc-900 text-zinc-800 text-xs font-medium px-2.5 py-1.5 rounded transition"
              >
                <Plus size={14} />
                <span>New Case</span>
              </button>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <a
              href={`/api/cases/${selectedCaseId}/export-script`}
              target="_blank"
              rel="noreferrer"
              className="flex items-center space-x-1.5 text-xs font-medium border border-zinc-300 hover:border-zinc-900 text-zinc-800 px-3 py-1.5 rounded transition"
            >
              <FileCode size={14} />
              <span>Export Script.py</span>
            </a>

            <button
              onClick={handleExecute}
              disabled={running}
              className={`flex items-center space-x-1.5 px-4 py-1.5 rounded text-xs font-semibold text-white transition ${
                running ? 'bg-zinc-400 cursor-not-allowed' : 'bg-zinc-950 hover:bg-zinc-800'
              }`}
            >
              {running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
              <span>{running ? 'Running Pipeline...' : 'Run Pipeline'}</span>
            </button>
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
                className={`flex items-center space-x-2 py-3 border-b-2 text-sm font-medium transition ${
                  active
                    ? 'border-zinc-950 text-zinc-950 font-semibold'
                    : 'border-transparent text-zinc-500 hover:text-zinc-900'
                }`}
              >
                <span className={`text-[11px] font-mono px-1.5 py-0.2 rounded ${active ? 'bg-zinc-900 text-white' : 'bg-zinc-100 text-zinc-600'}`}>
                  {tab.step}
                </span>
                <Icon size={16} />
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

                  <div className="text-sm font-mono font-bold text-white">
                    {caseDetail.job_state.progress}%
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
                      <div className="border border-zinc-200 rounded p-4 bg-zinc-50/50">
                        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Total Rows (Exact)</div>
                        <div className="text-2xl font-bold mt-1 text-zinc-950 font-mono">
                          {edaData.summary.total_rows.toLocaleString()}
                        </div>
                        <div className="text-xs text-zinc-500 mt-1">Full Training Records</div>
                      </div>

                      <div className="border border-zinc-200 rounded p-4 bg-zinc-50/50">
                        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Total Features</div>
                        <div className="text-2xl font-bold mt-1 text-zinc-950 font-mono">
                          {edaData.summary.total_cols} Cols
                        </div>
                        <div className="text-xs text-zinc-500 mt-1">
                          File Size: {edaData.summary.file_size_mb} MB
                        </div>
                      </div>

                      <div className="border border-zinc-200 rounded p-4 bg-zinc-50/50">
                        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Missing Values (Sample)</div>
                        <div className="text-2xl font-bold mt-1 text-zinc-950 font-mono">
                          {edaData.summary.missing_pct_sample}
                        </div>
                        <div className="text-xs text-zinc-500 mt-1">
                          {edaData.summary.missing_cells_sample.toLocaleString()} cells ({edaData.summary.null_cols_note || 'in sample'})
                        </div>
                      </div>

                      <div className="border border-zinc-200 rounded p-4 bg-zinc-50/50">
                        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Target Variable</div>
                        <div className="text-2xl font-bold mt-1 text-zinc-950 font-mono truncate">
                          {edaData.summary.target_col}
                        </div>
                        <div className="text-xs text-zinc-500 mt-1">
                          {edaData.target_distribution.length} Distinct Classes
                        </div>
                      </div>
                    </div>

                    {/* Target Distribution & Correlation Ranking */}
                    <div className="grid grid-cols-3 gap-8">
                      {/* Target Class Balance */}
                      <div className="border border-zinc-200 rounded p-5 bg-white space-y-4">
                        <h3 className="font-semibold text-sm text-zinc-950">Target Class Balance</h3>
                        <p className="text-xs text-zinc-500">
                          Class distribution in sample. Equal distribution indicates no class imbalance treatment needed.
                        </p>

                        <div className="space-y-3 pt-2">
                          {edaData.target_distribution.map(td => (
                            <div key={td.label} className="space-y-1">
                              <div className="flex justify-between text-xs font-medium">
                                <span className="font-mono text-zinc-900">{td.label}</span>
                                <span className="text-zinc-500 font-mono">{td.percentage}% ({td.count.toLocaleString()})</span>
                              </div>
                              <div className="w-full bg-zinc-100 h-2.5 rounded overflow-hidden">
                                <div className="bg-zinc-900 h-full rounded" style={{ width: `${td.percentage}%` }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Top Correlations with Target */}
                      <div className="col-span-2 border border-zinc-200 rounded p-5 bg-white space-y-4">
                        <div className="flex justify-between items-center">
                          <h3 className="font-semibold text-sm text-zinc-950">Top Predictive Signals (Correlations with Target)</h3>
                          <span className="text-xs font-mono text-zinc-500">Pearson Correlation (r)</span>
                        </div>
                        <p className="text-xs text-zinc-500">
                          Features with highest linear correlation magnitude with {edaData.summary.target_col}. Strong indicators for model splitting.
                        </p>

                        <div className="grid grid-cols-2 gap-x-6 gap-y-2 pt-2">
                          {edaData.correlations.map((c, idx) => (
                            <div 
                              key={c.feature} 
                              onClick={() => setSelectedFeature(c.feature)}
                              className="flex items-center justify-between text-xs font-mono py-1 border-b border-zinc-100 hover:bg-zinc-50 cursor-pointer px-1 rounded transition"
                              title="Click to view distribution"
                            >
                              <span className="truncate w-44 text-zinc-800 font-medium">#{idx + 1} {c.feature}</span>
                              <span className={`font-semibold ${c.correlation >= 0 ? 'text-zinc-950' : 'text-zinc-500'}`}>
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

                          {/* Distribution Bars */}
                          {activeCol?.distribution && activeCol.distribution.length > 0 ? (
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

                {/* 2. Custom Domain Hook (feature_hook.py) */}
                <div className="border border-zinc-200 rounded p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h4 className="font-semibold text-sm text-zinc-950">Registered Domain Hook: `feature_hook.py`</h4>
                      <p className="text-xs text-zinc-500">Loaded from cases/{selectedCaseId}/feature_hook.py (Domain physics & contextual imputations)</p>
                    </div>

                    <a
                      href={`/api/cases/${selectedCaseId}/export-script`}
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center space-x-1.5 border border-zinc-300 hover:border-zinc-900 text-xs font-semibold px-3 py-1.5 rounded transition"
                    >
                      <Download size={14} />
                      <span>Download Full Pipeline Script (.py)</span>
                    </a>
                  </div>

                  <pre className="bg-zinc-950 text-zinc-100 p-5 rounded font-mono text-xs overflow-x-auto leading-relaxed border border-zinc-800">
                    {caseDetail.hook_code || '# No custom feature hook registered. Universal Auto-FE primitives used.'}
                  </pre>
                </div>
              </div>
            )}

            {/* TAB 3: MODEL TRAINING & CV LEADERBOARD */}
            {activeTab === 'training' && (
              <div className="space-y-8">
                {/* 4 Status KPI Cards */}
                <div className="grid grid-cols-4 gap-4">
                  <div className="border border-zinc-200 rounded p-4 bg-zinc-50/50">
                    <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Target Variable</div>
                    <div className="text-xl font-bold mt-1 text-zinc-950">{meta?.target_col}</div>
                    <div className="text-xs text-zinc-500 mt-1 capitalize">{meta?.task_type} Classification</div>
                  </div>

                  <div className="border border-zinc-200 rounded p-4 bg-zinc-50/50">
                    <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Primary Metric</div>
                    <div className="text-xl font-bold mt-1 text-zinc-950">{meta?.eval_metric.toUpperCase()}</div>
                    <div className="text-xs text-zinc-500 mt-1">{meta?.n_splits}-Fold {meta?.cv_strategy}</div>
                  </div>

                  <div className="border border-zinc-200 rounded p-4 bg-zinc-50/50">
                    <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                      Best Single Model ({bestSingleName.toUpperCase() || 'N/A'})
                    </div>
                    <div className="text-xl font-bold mt-1 text-zinc-950">
                      {bestSingleScore ? bestSingleScore.toFixed(5) : 'Not Run'}
                    </div>
                    <div className="text-xs text-zinc-500 mt-1">Single Model Baseline</div>
                  </div>

                  <div className="border border-zinc-200 rounded p-4 bg-zinc-50/50">
                    <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Champion Ensemble</div>
                    <div className="text-xl font-bold mt-1 text-zinc-950 flex items-center space-x-2">
                      <span>{championScore ? championScore.toFixed(5) : 'Not Run'}</span>
                      {championScore && bestSingleScore && (
                        <span className="text-xs font-mono font-medium px-1.5 py-0.5 rounded bg-zinc-200 text-zinc-800">
                          +{ (championScore - bestSingleScore).toFixed(5) }
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-zinc-500 mt-1 font-mono truncate">{championModel || 'None'}</div>
                  </div>
                </div>

                {/* Optuna Bayesian Hyperparameter Optimization Studio */}
                <div className="border border-zinc-200 rounded p-6 bg-white space-y-5">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-100 pb-4">
                    <div>
                      <div className="flex items-center space-x-2">
                        <h3 className="font-bold text-sm text-zinc-950 uppercase tracking-tight">Automated Hyperparameter Optimization (Optuna Engine)</h3>
                        <span className="text-[10px] font-mono uppercase bg-zinc-900 text-white px-2 py-0.5 rounded">
                          Bayesian TPE
                        </span>
                      </div>
                      <p className="text-xs text-zinc-500 mt-1">
                        Discovers deeper tree architectures, learning rates, and L1/L2 regularization to maximize OOF score without manual trial-and-error.
                      </p>
                    </div>

                    {/* Tuning Action Controls */}
                    <div className="flex items-center space-x-3">
                      <select
                        value={tuneModel}
                        onChange={e => setTuneModel(e.target.value as any)}
                        disabled={tuning}
                        className="bg-zinc-50 border border-zinc-300 rounded px-3 py-1.5 text-xs font-mono font-medium text-zinc-900 focus:outline-none"
                      >
                        <option value="lgbm">LightGBM</option>
                        <option value="xgb">XGBoost</option>
                        <option value="catboost">CatBoost</option>
                      </select>

                      <select
                        value={tuneTrials}
                        onChange={e => setTuneTrials(Number(e.target.value))}
                        disabled={tuning}
                        className="bg-zinc-50 border border-zinc-300 rounded px-3 py-1.5 text-xs font-mono font-medium text-zinc-900 focus:outline-none"
                      >
                        <option value={10}>10 Trials (Fast)</option>
                        <option value={15}>15 Trials (Balanced)</option>
                        <option value={25}>25 Trials (Deep)</option>
                      </select>

                      <button
                        onClick={handleStartTune}
                        disabled={tuning || running}
                        className={`flex items-center space-x-1.5 px-4 py-1.5 rounded text-xs font-semibold text-white transition ${
                          tuning || running ? 'bg-zinc-400 cursor-not-allowed' : 'bg-zinc-950 hover:bg-zinc-800'
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
                      <div className="grid grid-cols-3 gap-4">
                        <div className="border border-zinc-200 rounded p-3 bg-zinc-50/50">
                          <div className="text-[10px] uppercase font-semibold text-zinc-500">Optimization Status</div>
                          <div className="text-sm font-bold text-zinc-900 mt-0.5 capitalize font-mono">{tuneStatus.status}</div>
                        </div>
                        <div className="border border-zinc-200 rounded p-3 bg-zinc-50/50">
                          <div className="text-[10px] uppercase font-semibold text-zinc-500">Trials Progress</div>
                          <div className="text-sm font-bold text-zinc-900 mt-0.5 font-mono">
                            {tuneStatus.current_trial || tuneStatus.trials?.length || 0} / {tuneStatus.total_trials || tuneTrials}
                          </div>
                        </div>
                        <div className="border border-zinc-200 rounded p-3 bg-zinc-50/50">
                          <div className="text-[10px] uppercase font-semibold text-zinc-500">Best Trial {meta?.eval_metric.toUpperCase()}</div>
                          <div className="text-sm font-bold text-zinc-950 mt-0.5 font-mono">
                            {tuneStatus.best_score ? tuneStatus.best_score.toFixed(5) : '-'}
                          </div>
                        </div>
                      </div>

                      {tuneStatus.best_params && (
                        <div className="border border-zinc-200 rounded p-4 bg-zinc-50/30 flex flex-col md:flex-row md:items-center justify-between gap-4">
                          <div className="space-y-1">
                            <div className="text-xs font-semibold text-zinc-950">Discovered Optimal Hyperparameters ({tuneModel.toUpperCase()})</div>
                            <pre className="text-[11px] font-mono text-zinc-600 bg-white p-2.5 rounded border border-zinc-200 max-h-32 overflow-y-auto">
                              {JSON.stringify(tuneStatus.best_params, null, 2)}
                            </pre>
                          </div>
                          <button
                            onClick={handleApplyTunedParams}
                            className="flex items-center space-x-1.5 bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-semibold px-4 py-2 rounded transition whitespace-nowrap self-start md:self-auto"
                          >
                            <Check size={14} />
                            <span>Apply to Pipeline Config</span>
                          </button>
                        </div>
                      )}

                      {/* Optuna Console Log */}
                      <div className="bg-zinc-950 text-zinc-100 rounded p-3 font-mono text-[11px] max-h-40 overflow-y-auto space-y-1">
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
                <label className="text-xs font-semibold text-zinc-700">Train CSV File Path</label>
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
                    onClick={handleInspect}
                    disabled={inspectLoading}
                    className="px-3 py-2 border border-zinc-300 hover:border-zinc-900 text-xs font-medium rounded whitespace-nowrap flex items-center space-x-1"
                  >
                    {inspectLoading && <Loader2 size={12} className="animate-spin" />}
                    <span>{inspectLoading ? 'Inspecting...' : 'Inspect'}</span>
                  </button>
                </div>
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
                <input
                  type="text"
                  placeholder="e.g. playground-series/dataset/test.csv"
                  value={newCase.test_path}
                  onChange={e => setNewCase({ ...newCase, test_path: e.target.value })}
                  className="mt-1 w-full text-xs border border-zinc-300 rounded p-2 font-mono text-zinc-900 focus:outline-none focus:border-zinc-950"
                />
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
