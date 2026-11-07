import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Download, 
  Database, 
  Loader2, 
  AlertTriangle, 
  RefreshCw, 
  BarChart2, 
  Cpu,
  Check,
  Zap,
  Square,
  Terminal,
  Copy,
  Sliders,
  Layers,
  TrendingUp,
  Sparkles
} from 'lucide-react';

interface TextStudioProps {
  selectedCaseId: string;
  caseDetail: any;
  onRefreshCase: () => void;
}

export const TextStudio: React.FC<TextStudioProps> = ({
  selectedCaseId,
  caseDetail,
  onRefreshCase
}) => {
  const [activeNlpTab, setActiveNlpTab] = useState<'preprocess' | 'eda' | 'baseline' | 'deep_nlp' | 'submission'>('preprocess');

  // Modular Text Cleaner & Field Constructor State
  const [selectedColumnsToCombine, setSelectedColumnsToCombine] = useState<string[]>([]);
  const [prefixWithTag, setPrefixWithTag] = useState<boolean>(true);
  const [targetCleanCol, setTargetCleanCol] = useState<string>('clean_text');
  
  // Cleaning Rules Toggles
  const [stripHtml, setStripHtml] = useState<boolean>(true);
  const [stripUrls, setStripUrls] = useState<boolean>(true);
  const [stripMentions, setStripMentions] = useState<boolean>(false);
  const [stripDateline, setStripDateline] = useState<boolean>(true);
  const [reduceRepeated, setReduceRepeated] = useState<boolean>(true);
  const [toLower, setToLower] = useState<boolean>(true);
  const [customRemovePhrases, setCustomRemovePhrases] = useState<string>('baca juga:\nsimak breaking news');
  const [customRegex, setCustomRegex] = useState<string>('');
  
  const [isCleaning, setIsCleaning] = useState<boolean>(false);
  const [cleanResult, setCleanResult] = useState<any>(null);

  // Text EDA State
  const [edaData, setEdaData] = useState<any>(null);
  const [selectedEdaCol, setSelectedEdaCol] = useState<string>('');
  const [edaLoading, setEdaLoading] = useState<boolean>(false);
  const [edaError, setEdaError] = useState<string | null>(null);

  // Baseline Modeling State
  const [baselineTextCol, setBaselineTextCol] = useState<string>('');
  const [baselineLabelCol, setBaselineLabelCol] = useState<string>('');
  const [maxFeatures, setMaxFeatures] = useState<number>(15000);
  const [ngramRange, setNgramRange] = useState<string>('1-2');
  const [nSplits, setNSplits] = useState<number>(5);
  const [regC, setRegC] = useState<number>(1.5);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [baselineResult, setBaselineResult] = useState<any>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  const availableCols: string[] = caseDetail?.dataset_columns || [];

  // Initialize columns when caseDetail loads
  useEffect(() => {
    if (caseDetail) {
      const meta = caseDetail.meta || {};
      const activeTextCol = meta.text_column || (availableCols.includes('clean_text') ? 'clean_text' : (availableCols.includes('content') ? 'content' : (availableCols.includes('text') ? 'text' : availableCols[0] || '')));
      setBaselineTextCol(activeTextCol);
      setBaselineLabelCol(meta.target_column || meta.target_col || (availableCols.includes('category') ? 'category' : (availableCols.includes('label') ? 'label' : '')));
      
      // Auto-select text columns to combine if not set
      if (availableCols.length > 0 && selectedColumnsToCombine.length === 0) {
        const textLike = availableCols.filter(c => {
          const l = c.toLowerCase();
          return l === 'title' || l === 'content' || l === 'keywords' || l === 'body' || l === 'text' || l === 'headline';
        });
        setSelectedColumnsToCombine(textLike.length > 0 ? textLike : [availableCols[0]]);
      }
    }
  }, [caseDetail]);

  useEffect(() => {
    if (selectedCaseId) {
      fetchEda();
    }
  }, [selectedCaseId]);

  const fetchEda = async (columnToAnalyze?: string) => {
    if (!selectedCaseId) return;
    setEdaLoading(true);
    setEdaError(null);
    try {
      const trainPath = `cases/${selectedCaseId}/train.csv`;
      const targetCol = columnToAnalyze || selectedEdaCol || caseDetail?.meta?.text_column || null;
      const res = await fetch('/api/tools/text-eda', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: trainPath,
          text_col: targetCol,
          label_col: caseDetail?.meta?.target_column || caseDetail?.meta?.target_col || null
        })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to generate Text EDA');
      }
      const data = await res.json();
      setEdaData(data);
      if (data.text_col) {
        setSelectedEdaCol(data.text_col);
      }
    } catch (err: any) {
      setEdaError(err.message);
    } finally {
      setEdaLoading(false);
    }
  };

  const handleExecuteClean = async () => {
    setIsCleaning(true);
    try {
      const phrases = customRemovePhrases
        .split('\n')
        .map(s => s.trim())
        .filter(s => s.length > 0);

      const regexList = customRegex
        .split('\n')
        .map(s => s.trim())
        .filter(s => s.length > 0);

      const targetCol = targetCleanCol || 'clean_text';

      const res = await fetch(`/api/cases/${selectedCaseId}/nlp-clean`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          columns_to_combine: selectedColumnsToCombine,
          prefix_with_column_name: prefixWithTag,
          target_text_col: targetCol,
          lower: toLower,
          strip_urls: stripUrls,
          strip_mentions: stripMentions,
          strip_html: stripHtml,
          strip_dateline: stripDateline,
          reduce_repeated_chars: reduceRepeated,
          custom_remove_phrases: phrases.length > 0 ? phrases : null,
          custom_regex_list: regexList.length > 0 ? regexList : null
        })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Cleaning failed');
      }
      const data = await res.json();
      setCleanResult(data);
      setBaselineTextCol(targetCol);
      onRefreshCase();
      fetchEda(targetCol);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setIsCleaning(false);
    }
  };

  const handleTrainBaseline = async () => {
    setIsTraining(true);
    setTrainingError(null);
    try {
      const [nMin, nMax] = ngramRange.split('-').map(Number);
      const res = await fetch(`/api/cases/${selectedCaseId}/nlp-train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text_col: baselineTextCol || caseDetail?.meta?.text_column || 'clean_text',
          label_col: baselineLabelCol || caseDetail?.meta?.target_column || 'category',
          id_col: caseDetail?.meta?.id_column || 'id',
          max_features: maxFeatures,
          ngram_min: nMin || 1,
          ngram_max: nMax || 2,
          n_splits: nSplits,
          c_param: regC
        })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Training failed');
      }
      const data = await res.json();
      setBaselineResult(data);
      onRefreshCase();
    } catch (err: any) {
      setTrainingError(err.message);
    } finally {
      setIsTraining(false);
    }
  };

  // Transformer Pipeline State
  const [transformerModel, setTransformerModel] = useState<string>('indobenchmark/indobert-base-p1');
  const [transformerEpochs, setTransformerEpochs] = useState<number>(3);
  const [transformerBatchSize, setTransformerBatchSize] = useState<number>(16);
  const [transformerLr, setTransformerLr] = useState<number>(2e-5);
  const [transformerMaxLen, setTransformerMaxLen] = useState<string>('auto');
  const [transformerStatus, setTransformerStatus] = useState<any>(null);
  const [isStartingTransformer, setIsStartingTransformer] = useState<boolean>(false);
  const [copiedCli, setCopiedCli] = useState<boolean>(false);

  // Ensemble & Blending State
  const [ensembleBaselineWeight, setEnsembleBaselineWeight] = useState<number>(30); // 30% baseline, 70% transformer
  const [isBlending, setIsBlending] = useState<boolean>(false);
  const [ensembleResult, setEnsembleResult] = useState<any>(null);
  const [ensembleError, setEnsembleError] = useState<string | null>(null);

  const fetchEnsembleStatus = async () => {
    if (!selectedCaseId) return;
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/nlp-ensemble-status`);
      if (res.ok) {
        const data = await res.json();
        if (data.ensemble_macro_f1 || data.optimal_baseline_weight !== undefined) {
          setEnsembleResult(data);
          if (data.optimal_baseline_weight !== undefined) {
            setEnsembleBaselineWeight(Math.round(data.optimal_baseline_weight * 100));
          }
        }
      }
    } catch (e) {
      // ignore
    }
  };

  useEffect(() => {
    if (activeNlpTab === 'submission') {
      fetchEnsembleStatus();
    }
  }, [activeNlpTab, selectedCaseId]);

  const handleRunEnsemble = async (autoOptimize: boolean = true) => {
    setIsBlending(true);
    setEnsembleError(null);
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/nlp-ensemble`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          baseline_weight: autoOptimize ? undefined : ensembleBaselineWeight / 100,
          auto_optimize: autoOptimize
        })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Ensemble blending failed');
      }
      const data = await res.json();
      setEnsembleResult(data);
      if (data.optimal_baseline_weight !== undefined) {
        setEnsembleBaselineWeight(Math.round(data.optimal_baseline_weight * 100));
      }
    } catch (e: any) {
      setEnsembleError(e.message);
    } finally {
      setIsBlending(false);
    }
  };

  const fetchTransformerStatus = async () => {
    if (!selectedCaseId) return;
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/nlp-transformer-status`);
      if (res.ok) {
        const data = await res.json();
        setTransformerStatus(data);
      }
    } catch (e) {
      // ignore
    }
  };

  useEffect(() => {
    if (activeNlpTab === 'deep_nlp') {
      fetchTransformerStatus();
    }
  }, [activeNlpTab, selectedCaseId]);

  useEffect(() => {
    let interval: any = null;
    if (activeNlpTab === 'deep_nlp' && transformerStatus?.is_running) {
      interval = setInterval(() => {
        fetchTransformerStatus();
      }, 2500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [activeNlpTab, transformerStatus?.is_running]);

  const handleLaunchTransformer = async () => {
    setIsStartingTransformer(true);
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/nlp-train-transformer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: transformerModel,
          epochs: transformerEpochs,
          batch_size: transformerBatchSize,
          learning_rate: transformerLr,
          max_length: transformerMaxLen === 'auto' ? null : Number(transformerMaxLen)
        })
      });
      if (res.ok) {
        await fetchTransformerStatus();
      }
    } catch (e) {
      // ignore
    } finally {
      setIsStartingTransformer(false);
    }
  };

  const handleStopTransformer = async () => {
    try {
      await fetch(`/api/cases/${selectedCaseId}/nlp-transformer-stop`, { method: 'POST' });
      await fetchTransformerStatus();
    } catch (e) {
      // ignore
    }
  };

  const handleCopyCli = () => {
    const cliCmd = `python train_transformer_universal.py --case ${selectedCaseId}`;
    navigator.clipboard.writeText(cliCmd);
    setCopiedCli(true);
    setTimeout(() => setCopiedCli(false), 2000);
  };

  const toggleColumnSelection = (col: string) => {
    setSelectedColumnsToCombine(prev => 
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    );
  };

  return (
    <div className="space-y-6">
      {/* Case Header & Quick Metrics */}
      <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold uppercase tracking-wider bg-zinc-100 text-zinc-700 border border-zinc-200">
                Text Mining & NLP Studio
              </span>
            </div>
            <h2 className="text-lg font-bold text-zinc-950 mt-1">
              {caseDetail?.meta?.name || selectedCaseId}
            </h2>
            <p className="text-xs text-zinc-500 mt-0.5 font-mono">
              Target: <span className="text-zinc-900 font-semibold">{caseDetail?.meta?.target_column || caseDetail?.meta?.target_col || 'category'}</span> | 
              Text: <span className="text-zinc-900 font-semibold">{caseDetail?.meta?.text_column || baselineTextCol || 'clean_text'}</span> |
              Task: <span className="text-zinc-900 font-semibold">{caseDetail?.meta?.task_type || 'multiclass'}</span>
            </p>
          </div>

          {/* Quick Metrics from EDA */}
          {edaData && (
            <div className="flex items-center space-x-5 border-l border-zinc-100 pl-5">
              <div>
                <div className="text-[10px] font-mono uppercase text-zinc-400">Documents</div>
                <div className="text-sm font-bold text-zinc-900">{edaData.total_documents?.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-[10px] font-mono uppercase text-zinc-400">Vocabulary</div>
                <div className="text-sm font-bold text-zinc-900">{edaData.vocabulary_size?.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-[10px] font-mono uppercase text-zinc-400">Median Words</div>
                <div className="text-sm font-bold text-zinc-900">{edaData.word_length_stats?.median || 0}</div>
              </div>
              <div>
                <div className="text-[10px] font-mono uppercase text-zinc-400">P95 Length</div>
                <div className="text-sm font-bold text-zinc-900">{edaData.word_length_stats?.p95 || 0}</div>
              </div>
            </div>
          )}
        </div>

        {/* NLP Sub-Navigation Tabs */}
        <div className="flex items-center space-x-1 border-t border-zinc-100 mt-5 pt-3">
          {[
            { id: 'preprocess', label: '1. Preprocessing & Field Fusion', icon: RefreshCw },
            { id: 'eda', label: '2. Corpus & Vocab EDA', icon: BarChart2 },
            { id: 'baseline', label: '3. TF-IDF & Baseline CV', icon: Play },
            { id: 'deep_nlp', label: '4. Deep Learning (Transformers)', icon: Cpu },
            { id: 'submission', label: '5. Submissions & Inference', icon: Download },
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = activeNlpTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveNlpTab(tab.id as any)}
                className={`flex items-center space-x-2 px-3.5 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                  isActive
                    ? 'bg-zinc-900 text-white shadow-xs'
                    : 'text-zinc-600 hover:text-zinc-950 hover:bg-zinc-100'
                }`}
              >
                <Icon size={13} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab 1: Preprocessing & Field Fusion */}
      {activeNlpTab === 'preprocess' && (
        <div className="space-y-6">
          {/* Top: Dataset Schema Card */}
          <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Database size={15} className="text-zinc-600" />
                <h3 className="text-sm font-bold text-zinc-900">Ingested Dataset Schema</h3>
              </div>
              <span className="text-[11px] font-mono bg-zinc-100 text-zinc-700 px-2.5 py-0.5 rounded border border-zinc-200">
                {availableCols.length} Columns Detected
              </span>
            </div>

            <p className="text-xs text-zinc-500 mb-3">
              Dataset has been normalized and is ready for text preprocessing. Select fields below to combine into your model input text.
            </p>

            <div className="flex flex-wrap gap-2">
              {availableCols.map(col => (
                <div key={col} className="flex items-center space-x-1.5 bg-zinc-50 border border-zinc-200 px-2.5 py-1 rounded text-xs font-mono text-zinc-800">
                  <span className="font-semibold">{col}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Main: Modular Text Cleaning & Field Fusion Card */}
          <div className="bg-white border border-zinc-200/80 rounded-xl p-6 shadow-xs space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-zinc-900">Text Cleaning & Field Fusion</h3>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Combine informative fields (e.g. title + keywords + content) and strip editorial boilerplate to create your final model text.
                </p>
              </div>
              <span className="text-[10px] font-mono bg-zinc-100 text-zinc-600 px-2 py-0.5 rounded border border-zinc-200">
                Domain-Agnostic
              </span>
            </div>

            <div className="space-y-4 pt-2">
              {/* Step A: Column Selection */}
              <div>
                <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1.5">
                  1. Select Columns to Combine
                </label>
                <div className="flex flex-wrap gap-2">
                  {availableCols.map(col => {
                    const isSelected = selectedColumnsToCombine.includes(col);
                    return (
                      <button
                        type="button"
                        key={col}
                        onClick={() => toggleColumnSelection(col)}
                        className={`px-3 py-1.5 text-xs font-mono rounded-lg border transition ${
                          isSelected
                            ? 'bg-zinc-900 text-white border-zinc-900 font-semibold shadow-xs'
                            : 'bg-zinc-50 text-zinc-700 border-zinc-200 hover:border-zinc-300'
                        }`}
                      >
                        {isSelected ? '✓ ' : '+ '}{col}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Tag prefix & Target Column Name */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center bg-zinc-50/60 p-4 rounded-xl border border-zinc-200/60">
                <div>
                  <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">Target Column Name</label>
                  <input
                    type="text"
                    value={targetCleanCol}
                    onChange={e => setTargetCleanCol(e.target.value)}
                    className="w-full text-xs font-mono bg-white border border-zinc-200 rounded-lg px-3 py-2 text-zinc-900"
                    placeholder="clean_text"
                  />
                </div>
                <div className="pt-2">
                  <label className="flex items-center space-x-2 text-xs text-zinc-800 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={prefixWithTag}
                      onChange={e => setPrefixWithTag(e.target.checked)}
                      className="rounded border-zinc-300 text-zinc-900 focus:ring-zinc-950"
                    />
                    <span>Prefix fields with structural tags: <code className="font-mono bg-white px-1 py-0.5 rounded text-[11px] border border-zinc-200">[TITLE] ... [CONTENT] ...</code></span>
                  </label>
                </div>
              </div>

              {/* Step B: Cleaning Rules */}
              <div>
                <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1.5">
                  2. Cleaning Rules
                </label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5 text-xs text-zinc-700 bg-zinc-50/60 p-4 rounded-xl border border-zinc-200/60">
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={stripHtml}
                      onChange={e => setStripHtml(e.target.checked)}
                      className="rounded border-zinc-300"
                    />
                    <span>Strip HTML tags</span>
                  </label>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={stripUrls}
                      onChange={e => setStripUrls(e.target.checked)}
                      className="rounded border-zinc-300"
                    />
                    <span>Strip URLs</span>
                  </label>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={stripDateline}
                      onChange={e => setStripDateline(e.target.checked)}
                      className="rounded border-zinc-300"
                    />
                    <span>Strip News Datelines (CITY, Media -)</span>
                  </label>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={stripMentions}
                      onChange={e => setStripMentions(e.target.checked)}
                      className="rounded border-zinc-300"
                    />
                    <span>Strip @User Mentions</span>
                  </label>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={reduceRepeated}
                      onChange={e => setReduceRepeated(e.target.checked)}
                      className="rounded border-zinc-300"
                    />
                    <span>Normalize repeated letters</span>
                  </label>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={toLower}
                      onChange={e => setToLower(e.target.checked)}
                      className="rounded border-zinc-300"
                    />
                    <span>Convert to lowercase</span>
                  </label>
                </div>
              </div>

              {/* Step C: Custom Removal Phrases & Regex */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">
                    3. Custom Phrases to Remove (one per line)
                  </label>
                  <textarea
                    rows={2}
                    value={customRemovePhrases}
                    onChange={e => setCustomRemovePhrases(e.target.value)}
                    placeholder="e.g.&#10;baca juga:&#10;simak breaking news"
                    className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg p-2.5 text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-950"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">
                    Custom Regex Patterns (Optional, one per line)
                  </label>
                  <textarea
                    rows={2}
                    value={customRegex}
                    onChange={e => setCustomRegex(e.target.value)}
                    placeholder="e.g.&#10;^editor\s*:.*&#10;\[advertisement\]"
                    className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg p-2.5 text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-950"
                  />
                </div>
              </div>

              <button
                type="button"
                onClick={handleExecuteClean}
                disabled={isCleaning || selectedColumnsToCombine.length === 0}
                className="w-full flex items-center justify-center space-x-2 bg-zinc-900 hover:bg-zinc-800 disabled:bg-zinc-200 text-white text-xs font-medium py-3 rounded-xl transition shadow-xs"
              >
                {isCleaning ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                <span>{isCleaning ? 'Cleaning & Generating Column...' : 'Apply Cleaner & Generate Text'}</span>
              </button>
            </div>

            {cleanResult && (
              <div className="p-4 bg-emerald-50/50 border border-emerald-200/80 rounded-xl text-xs space-y-2">
                <div className="flex items-center space-x-2 text-emerald-800 font-semibold">
                  <Check size={14} className="text-emerald-600" />
                  <span>Column '{cleanResult.text_column}' Generated Successfully ({cleanResult.total_rows?.toLocaleString()} rows)</span>
                </div>
                <div className="text-[11px] font-mono text-zinc-700 bg-white p-3 rounded-lg border border-zinc-200 max-h-24 overflow-y-auto">
                  {cleanResult.sample_cleaned?.[0] || 'No sample'}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab 2: Corpus & Vocab EDA */}
      {activeNlpTab === 'eda' && (
        <div className="space-y-6">
          {/* Header Bar with Text Column Selector */}
          <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-zinc-900">Corpus Statistical Profile</h3>
              <p className="text-xs text-zinc-500">Distribution, word lengths, vocabulary richness, and class imbalance.</p>
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="text-xs text-zinc-500 whitespace-nowrap">Text Column to Analyze:</span>
              <select
                value={selectedEdaCol}
                onChange={e => {
                  setSelectedEdaCol(e.target.value);
                  fetchEda(e.target.value);
                }}
                className="text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg px-3 py-1.5 text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-950"
              >
                {availableCols.map(col => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {edaLoading && (
            <div className="flex items-center justify-center py-12 text-zinc-500 text-xs">
              <Loader2 size={16} className="animate-spin mr-2" />
              Generating Text EDA report...
            </div>
          )}

          {edaError && (
            <div className="p-4 bg-zinc-100 border border-zinc-200 rounded-xl text-xs text-zinc-700">
              {edaError}
            </div>
          )}

          {edaData && (
            <>
              {/* Dynamic Actionable Insights */}
              {edaData.actionable_insights && edaData.actionable_insights.length > 0 && (
                <div className="bg-emerald-50/60 border border-emerald-200/90 rounded-xl p-5 shadow-xs">
                  <div className="flex items-center space-x-2 mb-3">
                    <Sparkles size={16} className="text-emerald-700" />
                    <h3 className="text-sm font-bold text-emerald-950">
                      Automated NLP Modeling Insights & Recommendations
                    </h3>
                  </div>
                  <ul className="space-y-2">
                    {edaData.actionable_insights.map((insight: string, idx: number) => (
                      <li key={idx} className="flex items-start space-x-2 text-xs text-emerald-900 leading-relaxed">
                        <span className="text-emerald-600 font-bold mt-0.5">•</span>
                        <span>{insight}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Length Percentiles & Quality Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs">
                  <div className="text-[10px] font-mono uppercase text-zinc-400">Total Documents</div>
                  <div className="text-xl font-bold text-zinc-900 mt-1">{edaData.total_documents?.toLocaleString()}</div>
                  <div className="text-[11px] text-zinc-500 mt-1">Duplicates: {edaData.quality_metrics?.duplicate_documents} ({edaData.quality_metrics?.duplicate_percentage}%)</div>
                </div>
                <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs">
                  <div className="text-[10px] font-mono uppercase text-zinc-400">Vocabulary Size</div>
                  <div className="text-xl font-bold text-zinc-900 mt-1">{edaData.vocabulary_size?.toLocaleString()}</div>
                  <div className="text-[11px] text-zinc-500 mt-1">Lexical Richness: {edaData.lexical_richness}</div>
                </div>
                <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs">
                  <div className="text-[10px] font-mono uppercase text-zinc-400">Word Length Stats</div>
                  <div className="text-xl font-bold text-zinc-900 mt-1">{edaData.word_length_stats?.median} median</div>
                  <div className="text-[11px] text-zinc-500 mt-1">Min: {edaData.word_length_stats?.min} | Max: {edaData.word_length_stats?.max} | P95: {edaData.word_length_stats?.p95}</div>
                </div>
                <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs">
                  <div className="text-[10px] font-mono uppercase text-zinc-400">Char Length Stats</div>
                  <div className="text-xl font-bold text-zinc-900 mt-1">{edaData.char_length_stats?.median} median</div>
                  <div className="text-[11px] text-zinc-500 mt-1">P95: {edaData.char_length_stats?.p95?.toLocaleString()} chars</div>
                </div>
              </div>

              {/* Word Length Distribution Histogram (SVG Chart) */}
              {edaData.length_histogram && edaData.length_histogram.length > 0 && (() => {
                const hist = edaData.length_histogram;
                const maxCount = Math.max(...hist.map((b: any) => b.count), 1);
                const p95Val = edaData.word_length_stats?.p95 || 0;
                const maxBinEnd = hist[hist.length - 1]?.bin_end || 1;
                const p95Pct = Math.min(Math.max((p95Val / maxBinEnd), 0.05), 0.95);
                const p95X = 40 + p95Pct * 660;

                return (
                  <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-sm font-bold text-zinc-900">Word Count Distribution (Histogram)</h3>
                        <p className="text-xs text-zinc-500 mt-0.5">Frequency across 10 word-length intervals with P95 truncation cutoff</p>
                      </div>
                      <div className="flex items-center space-x-3 text-xs font-mono">
                        <span className="flex items-center space-x-1">
                          <span className="w-3 h-3 bg-zinc-800 rounded-sm inline-block"></span>
                          <span className="text-zinc-600">Sample Count</span>
                        </span>
                        <span className="flex items-center space-x-1">
                          <span className="w-3 h-0.5 bg-rose-500 inline-block"></span>
                          <span className="text-rose-600 font-semibold">P95 Threshold ({p95Val} words)</span>
                        </span>
                      </div>
                    </div>

                    <div className="w-full overflow-x-auto">
                      <svg viewBox="0 0 740 180" className="w-full h-44">
                        <line x1="40" y1="20" x2="700" y2="20" stroke="#F4F4F5" strokeWidth="1" />
                        <line x1="40" y1="75" x2="700" y2="75" stroke="#F4F4F5" strokeWidth="1" />
                        <line x1="40" y1="130" x2="700" y2="130" stroke="#E4E4E7" strokeWidth="1" />

                        {hist.map((b: any, i: number) => {
                          const barW = 52;
                          const x = 50 + i * 64;
                          const barH = Math.max(4, Math.round((b.count / maxCount) * 110));
                          const y = 130 - barH;

                          return (
                            <g key={i} className="group cursor-pointer">
                              <rect
                                x={x}
                                y={y}
                                width={barW}
                                height={barH}
                                rx="3"
                                className="fill-zinc-800 hover:fill-zinc-950 transition-colors"
                              />
                              <text
                                x={x + barW / 2}
                                y={y - 5}
                                textAnchor="middle"
                                className="text-[10px] font-mono fill-zinc-600 group-hover:fill-zinc-950 font-medium"
                              >
                                {b.pct}%
                              </text>
                              <text
                                x={x + barW / 2}
                                y="148"
                                textAnchor="middle"
                                className="text-[9px] font-mono fill-zinc-400"
                              >
                                {b.bin_start}-{b.bin_end}
                              </text>
                              <title>{`${b.bin_start} - ${b.bin_end} words: ${b.count?.toLocaleString()} docs (${b.pct}%)`}</title>
                            </g>
                          );
                        })}

                        <line
                          x1={p95X}
                          y1="15"
                          x2={p95X}
                          y2="130"
                          stroke="#F43F5E"
                          strokeWidth="2"
                          strokeDasharray="4 3"
                        />
                        <text
                          x={p95X}
                          y="12"
                          textAnchor="middle"
                          className="text-[10px] font-mono fill-rose-600 font-bold"
                        >
                          P95: {p95Val}w
                        </text>
                      </svg>
                    </div>
                  </div>
                );
              })()}

              {/* Class-Conditional Text Length Profile */}
              {edaData.class_length_profile && edaData.class_length_profile.length > 0 && (() => {
                const profiles = edaData.class_length_profile;
                const maxMean = Math.max(...profiles.map((p: any) => p.mean_words), 1);

                return (
                  <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                    <h3 className="text-sm font-bold text-zinc-900 mb-1">Class-Conditional Length Disparity</h3>
                    <p className="text-xs text-zinc-500 mb-4">Comparison of average document length across classes (reveals stylometric predictive signals)</p>

                    <div className="space-y-3">
                      {profiles.map((p: any, idx: number) => {
                        const pct = Math.max(6, Math.round((p.mean_words / maxMean) * 100));
                        return (
                          <div key={idx} className="space-y-1.5">
                            <div className="flex items-center justify-between text-xs">
                              <div className="flex items-center space-x-2">
                                <span className="font-semibold text-zinc-900">{p.class}</span>
                                <span className="text-[11px] font-mono text-zinc-400">({p.count} docs)</span>
                              </div>
                              <span className="font-mono text-xs text-zinc-700">
                                Mean: <strong className="text-zinc-950">{p.mean_words}</strong> words | Median: <strong className="text-zinc-950">{p.median_words}</strong> words
                              </span>
                            </div>
                            <div className="w-full bg-zinc-100 rounded-full h-2.5 overflow-hidden">
                              <div
                                className="bg-zinc-800 h-full rounded-full transition-all duration-500"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })()}

              {/* Class Imbalance Warning & Badges */}
              {edaData.class_distribution && Object.keys(edaData.class_distribution).length > 0 && (
                <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-zinc-900">
                      Class Distribution ({Object.keys(edaData.class_distribution).length} Classes)
                    </h3>
                    {Object.values(edaData.class_distribution).some((v: any) => v < 5) && (
                      <span className="flex items-center space-x-1 text-[11px] font-mono bg-zinc-100 text-zinc-800 px-2 py-0.5 rounded border border-zinc-200">
                        <AlertTriangle size={12} className="text-zinc-600" />
                        <span>Rare classes detected (&lt; 5 samples)</span>
                      </span>
                    )}
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {Object.entries(edaData.class_distribution).map(([cat, count]: [string, any]) => {
                      const isRare = count < 5;
                      return (
                        <div
                          key={cat}
                          className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-xs border ${
                            isRare
                              ? 'bg-zinc-100 border-zinc-300 text-zinc-700'
                              : 'bg-zinc-50 border-zinc-200 text-zinc-900'
                          }`}
                        >
                          <span className="font-medium">{cat}</span>
                          <span className="font-mono text-[11px] text-zinc-400">({count})</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Top Unigrams & Bigrams with Relative Frequency Progress Tracks */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                  <h3 className="text-sm font-bold text-zinc-900 mb-3">Top 20 Most Frequent Words</h3>
                  <div className="space-y-2">
                    {(() => {
                      const maxW = Math.max(...(edaData.top_words || []).map((w: any) => w.count), 1);
                      return edaData.top_words?.map((w: any, idx: number) => {
                        const pct = Math.max(4, Math.round((w.count / maxW) * 100));
                        return (
                          <div key={idx} className="flex items-center justify-between text-xs py-1 border-b border-zinc-100 last:border-0">
                            <div className="flex items-center space-x-3 flex-1 mr-4">
                              <span className="font-mono text-zinc-800 w-24 truncate font-medium">{w.word}</span>
                              <div className="flex-1 bg-zinc-100 rounded-full h-1.5 overflow-hidden">
                                <div className="bg-zinc-700 h-full rounded-full" style={{ width: `${pct}%` }} />
                              </div>
                            </div>
                            <span className="font-mono text-[11px] text-zinc-500 whitespace-nowrap">{w.count?.toLocaleString()}</span>
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>

                <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                  <h3 className="text-sm font-bold text-zinc-900 mb-3">Top 15 Most Frequent Bigrams</h3>
                  <div className="space-y-2">
                    {(() => {
                      const maxB = Math.max(...(edaData.top_bigrams || []).map((b: any) => b.count), 1);
                      return edaData.top_bigrams?.map((b: any, idx: number) => {
                        const pct = Math.max(4, Math.round((b.count / maxB) * 100));
                        return (
                          <div key={idx} className="flex items-center justify-between text-xs py-1 border-b border-zinc-100 last:border-0">
                            <div className="flex items-center space-x-3 flex-1 mr-4">
                              <span className="font-mono text-zinc-800 w-32 truncate font-medium">{b.bigram}</span>
                              <div className="flex-1 bg-zinc-100 rounded-full h-1.5 overflow-hidden">
                                <div className="bg-zinc-700 h-full rounded-full" style={{ width: `${pct}%` }} />
                              </div>
                            </div>
                            <span className="font-mono text-[11px] text-zinc-500 whitespace-nowrap">{b.count?.toLocaleString()}</span>
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Tab 3: TF-IDF & Baseline CV */}
      {activeNlpTab === 'baseline' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Configuration Column */}
            <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-zinc-900">Baseline Engine</h3>
                <span className="text-[10px] font-mono bg-zinc-100 text-zinc-600 px-2 py-0.5 rounded border border-zinc-200">
                  TF-IDF + Balanced LR
                </span>
              </div>

              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">Text Col</label>
                    <select
                      value={baselineTextCol}
                      onChange={e => setBaselineTextCol(e.target.value)}
                      className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg px-2 py-1.5 text-zinc-900"
                    >
                      {availableCols.map(c => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">Label Col</label>
                    <select
                      value={baselineLabelCol}
                      onChange={e => setBaselineLabelCol(e.target.value)}
                      className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg px-2 py-1.5 text-zinc-900"
                    >
                      {availableCols.map(c => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">Max Features</label>
                  <select
                    value={maxFeatures}
                    onChange={e => setMaxFeatures(Number(e.target.value))}
                    className="w-full text-xs bg-zinc-50 border border-zinc-200 rounded-lg px-3 py-2 text-zinc-900"
                  >
                    <option value={10000}>10,000 features</option>
                    <option value={15000}>15,000 features</option>
                    <option value={25000}>25,000 features</option>
                    <option value={50000}>50,000 features</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">N-Gram Range</label>
                  <select
                    value={ngramRange}
                    onChange={e => setNgramRange(e.target.value)}
                    className="w-full text-xs bg-zinc-50 border border-zinc-200 rounded-lg px-3 py-2 text-zinc-900"
                  >
                    <option value="1-1">Unigrams Only (1, 1)</option>
                    <option value="1-2">Unigrams + Bigrams (1, 2)</option>
                    <option value="1-3">Up to Trigrams (1, 3)</option>
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">CV Folds</label>
                    <input
                      type="number"
                      value={nSplits}
                      onChange={e => setNSplits(Number(e.target.value))}
                      className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg px-3 py-1.5 text-zinc-900"
                      min={2}
                      max={10}
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">Reg C</label>
                    <input
                      type="number"
                      value={regC}
                      onChange={e => setRegC(Number(e.target.value))}
                      className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg px-3 py-1.5 text-zinc-900"
                      step={0.1}
                    />
                  </div>
                </div>

                <button
                  type="button"
                  onClick={handleTrainBaseline}
                  disabled={isTraining}
                  className="w-full mt-3 flex items-center justify-center space-x-2 bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-medium py-2.5 rounded-lg transition"
                >
                  {isTraining ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
                  <span>{isTraining ? 'Training Cross-Validation...' : 'Train Stratified Baseline'}</span>
                </button>
              </div>

              {trainingError && (
                <div className="p-3 bg-zinc-100 border border-zinc-200 rounded-lg text-xs text-zinc-700">
                  {trainingError}
                </div>
              )}
            </div>

            {/* Results Column */}
            <div className="md:col-span-2 space-y-4">
              {baselineResult ? (
                <>
                  {/* Top Scores Banner */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                      <div className="text-[10px] font-mono uppercase text-zinc-400">Overall Accuracy</div>
                      <div className="text-3xl font-extrabold text-zinc-950 mt-1">
                        {(baselineResult.overall_accuracy * 100).toFixed(2)}%
                      </div>
                      <div className="text-xs text-zinc-500 mt-1">Cross-Validation Out-Of-Fold</div>
                    </div>
                    <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                      <div className="text-[10px] font-mono uppercase text-zinc-400">Macro-F1 Score</div>
                      <div className="text-3xl font-extrabold text-zinc-950 mt-1">
                        {(baselineResult.overall_macro_f1 * 100).toFixed(2)}%
                      </div>
                      <div className="text-xs text-zinc-500 mt-1">Rare Classes Shielded: {baselineResult.rare_classes_shielded}</div>
                    </div>
                  </div>

                  {/* Fold Results Breakdown */}
                  <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                    <h3 className="text-sm font-bold text-zinc-900 mb-3">Fold-by-Fold Validation</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left">
                        <thead>
                          <tr className="border-b border-zinc-200 text-zinc-400 font-mono text-[10px] uppercase">
                            <th className="pb-2">Fold</th>
                            <th className="pb-2">Accuracy</th>
                            <th className="pb-2">Macro-F1</th>
                            <th className="pb-2">Validation Samples</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-100">
                          {baselineResult.fold_results?.map((fr: any) => (
                            <tr key={fr.fold}>
                              <td className="py-2 font-mono font-medium text-zinc-900">Fold {fr.fold}</td>
                              <td className="py-2 font-mono text-zinc-700">{(fr.accuracy * 100).toFixed(2)}%</td>
                              <td className="py-2 font-mono text-zinc-700">{(fr.macro_f1 * 100).toFixed(2)}%</td>
                              <td className="py-2 font-mono text-zinc-400">{fr.val_samples}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Informative Terms per Class */}
                  {baselineResult.top_words_per_class && (
                    <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                      <h3 className="text-sm font-bold text-zinc-900 mb-3">Top Informative Terms per Class</h3>
                      <div className="space-y-2 max-h-80 overflow-y-auto pr-2">
                        {Object.entries(baselineResult.top_words_per_class).map(([cls, words]: [string, any]) => (
                          <div key={cls} className="text-xs border-b border-zinc-100 pb-2 last:border-0">
                            <span className="font-semibold text-zinc-900 mr-2">{cls}:</span>
                            <span className="font-mono text-zinc-600">{words.slice(0, 8).join(', ')}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="bg-white border border-zinc-200/80 rounded-xl p-12 text-center shadow-xs">
                  <Play size={24} className="mx-auto text-zinc-400 mb-3" />
                  <h4 className="text-sm font-bold text-zinc-900">No Baseline Trained Yet</h4>
                  <p className="text-xs text-zinc-500 mt-1 max-w-sm mx-auto">
                    Select columns and click "Train Stratified Baseline" to extract TF-IDF features and execute cross-validation.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Deep Learning (Transformers) */}
      {activeNlpTab === 'deep_nlp' && (
        <div className="space-y-6">
          {/* Header Card */}
          <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-[10px] font-mono uppercase bg-zinc-900 text-white px-2 py-0.5 rounded font-bold">
                  GPU Accelerated Fine-Tuning
                </span>
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded border uppercase font-bold ${
                  transformerStatus?.is_running 
                    ? 'bg-amber-50 text-amber-800 border-amber-300 animate-pulse'
                    : (transformerStatus?.state === 'completed' 
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-300' 
                        : 'bg-zinc-100 text-zinc-600 border-zinc-200')
                }`}>
                  {transformerStatus?.is_running ? 'Training in Progress (GPU Active)' : (transformerStatus?.state === 'completed' ? 'Model Ready & Validated' : 'Idle')}
                </span>
              </div>
              <h3 className="text-base font-bold text-zinc-950 mt-1">Deep Learning Transformers (IndoBERT / RoBERTa)</h3>
              <p className="text-xs text-zinc-500 mt-0.5">
                Otomatis mengadaptasi task ({caseDetail?.meta?.task_type || 'binary'}), panjang urutan teks (P95), dan bobot kelas berimbang.
              </p>
            </div>

            {/* Quick CLI Banner for PC Dosen */}
            <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-2.5 flex items-center space-x-3">
              <div className="font-mono text-[11px] text-zinc-700">
                <div className="text-[9px] uppercase tracking-wider text-zinc-400 font-sans font-semibold">PC Dosen / Terminal Execution:</div>
                <code>python train_transformer_universal.py --case {selectedCaseId}</code>
              </div>
              <button
                type="button"
                onClick={handleCopyCli}
                className="p-1.5 bg-white border border-zinc-200 rounded hover:bg-zinc-100 text-zinc-700 transition"
                title="Copy Terminal Command"
              >
                {copiedCli ? <Check size={13} className="text-emerald-600" /> : <Copy size={13} />}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Left Column: Hyperparameter & Launcher */}
            <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-zinc-100">
                <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-1.5">
                  <Zap size={13} className="text-zinc-900" />
                  <span>Pipeline Hyperparameters</span>
                </h4>
                <span className="text-[10px] font-mono text-zinc-500">PyTorch + FP16</span>
              </div>

              <div>
                <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">Backbone Pretrained Model</label>
                <select
                  value={transformerModel}
                  onChange={e => setTransformerModel(e.target.value)}
                  className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg px-2.5 py-2 text-zinc-900"
                >
                  <option value="indobenchmark/indobert-base-p1">indobenchmark/indobert-base-p1 (IndoBERT - Formal/News)</option>
                  <option value="indolem/indobertweet-base-uncased">indolem/indobertweet-base-uncased (IndoBERTweet - Slang/Comments)</option>
                  <option value="w11wo/indonesian-roberta-base-sentiment-classifier">w11wo/indonesian-roberta (Fine-tuned Sentiment)</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">Epochs</label>
                  <input
                    type="number"
                    value={transformerEpochs}
                    onChange={e => setTransformerEpochs(Number(e.target.value))}
                    className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg px-2.5 py-1.5 text-zinc-900"
                    min={1}
                    max={10}
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">Batch Size</label>
                  <select
                    value={transformerBatchSize}
                    onChange={e => setTransformerBatchSize(Number(e.target.value))}
                    className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg px-2.5 py-1.5 text-zinc-900"
                  >
                    <option value={8}>8 (Low VRAM)</option>
                    <option value={16}>16 (Recommended)</option>
                    <option value={32}>32 (Fast RTX 3070)</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">Learning Rate</label>
                  <input
                    type="number"
                    value={transformerLr}
                    onChange={e => setTransformerLr(Number(e.target.value))}
                    step={0.000005}
                    className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg px-2.5 py-1.5 text-zinc-900"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-mono uppercase text-zinc-500 mb-1">Max Sequence Length</label>
                  <select
                    value={transformerMaxLen}
                    onChange={e => setTransformerMaxLen(e.target.value)}
                    className="w-full text-xs font-mono bg-zinc-50 border border-zinc-200 rounded-lg px-2.5 py-1.5 text-zinc-900"
                  >
                    <option value="auto">Auto (from P95 = {edaData?.word_length_stats?.p95 ? Math.min(512, Math.max(96, Math.round(edaData.word_length_stats.p95 * 1.35))) : 128})</option>
                    <option value="96">96 tokens (Short/Fast)</option>
                    <option value="128">128 tokens (Comments)</option>
                    <option value="256">256 tokens (Articles)</option>
                    <option value="512">512 tokens (Full News)</option>
                  </select>
                </div>
              </div>

              <div className="pt-2">
                {transformerStatus?.is_running ? (
                  <button
                    type="button"
                    onClick={handleStopTransformer}
                    className="w-full flex items-center justify-center space-x-2 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold py-2.5 rounded-lg transition shadow-xs"
                  >
                    <Square size={13} />
                    <span>Hentikan Training GPU</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleLaunchTransformer}
                    disabled={isStartingTransformer}
                    className="w-full flex items-center justify-center space-x-2 bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-semibold py-2.5 rounded-lg transition shadow-xs disabled:opacity-50"
                  >
                    {isStartingTransformer ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
                    <span>Jalankan Fine-Tuning GPU (RTX 3070)</span>
                  </button>
                )}
              </div>

              <div className="text-[11px] text-zinc-500 space-y-1 bg-zinc-50 p-3 rounded-lg border border-zinc-200">
                <div className="font-semibold text-zinc-800">Spesifikasi Engine GPU:</div>
                <div>• Automatic FP16 mixed precision on CUDA</div>
                <div>• Rare class shielding to protect Stratified Split</div>
                <div>• Weighted CrossEntropyLoss anti class imbalance</div>
              </div>
            </div>

            {/* Right Column: Execution Monitor & Evaluation Results */}
            <div className="md:col-span-2 space-y-4">
              {/* If completed, show metric cards */}
              {transformerStatus?.results && (
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs">
                    <div className="text-[10px] font-mono uppercase text-zinc-400">Validation Macro-F1</div>
                    <div className="text-xl font-bold text-zinc-900 mt-0.5">
                      {(transformerStatus.results.macro_f1 * 100).toFixed(2)}%
                    </div>
                    {baselineResult?.overall_macro_f1 && (
                      <div className="text-[10px] font-mono text-emerald-600 mt-1 font-semibold">
                        + {((transformerStatus.results.macro_f1 - baselineResult.overall_macro_f1) * 100).toFixed(2)}% vs Baseline
                      </div>
                    )}
                  </div>
                  <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs">
                    <div className="text-[10px] font-mono uppercase text-zinc-400">Validation Accuracy</div>
                    <div className="text-xl font-bold text-zinc-900 mt-0.5">
                      {(transformerStatus.results.accuracy * 100).toFixed(2)}%
                    </div>
                    <div className="text-[10px] font-mono text-zinc-500 mt-1">
                      Loss: {Number(transformerStatus.results.eval_loss || 0).toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs">
                    <div className="text-[10px] font-mono uppercase text-zinc-400">Hardware & Backbone</div>
                    <div className="text-xs font-bold text-zinc-900 mt-1 truncate">
                      {transformerStatus.results.gpu_name || 'GPU CUDA'}
                    </div>
                    <div className="text-[10px] font-mono text-zinc-500 mt-1 truncate">
                      {transformerStatus.results.model_name?.split('/').pop()}
                    </div>
                  </div>
                </div>
              )}

              {/* Smart Loss & Sweet Spot Advisor Card */}
              {transformerStatus?.results?.smart_diagnostics && (
                <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-[10px] font-mono uppercase bg-zinc-900 text-white px-2 py-0.5 rounded font-bold">
                        Smart Loss & Sweet Spot Advisor
                      </span>
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded border uppercase font-bold ${
                        transformerStatus.results.smart_diagnostics.phase_color === 'emerald'
                          ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                          : (transformerStatus.results.smart_diagnostics.phase_color === 'blue'
                              ? 'bg-blue-50 text-blue-800 border-blue-300'
                              : (transformerStatus.results.smart_diagnostics.phase_color === 'amber'
                                  ? 'bg-amber-50 text-amber-800 border-amber-300'
                                  : 'bg-red-50 text-red-800 border-red-300'))
                      }`}>
                        {transformerStatus.results.smart_diagnostics.learning_phase}
                      </span>
                    </div>
                    <div className="text-[11px] font-mono text-zinc-500">
                      Loss Drop: <span className="font-bold text-zinc-900">-{transformerStatus.results.smart_diagnostics.loss_reduction_pct}%</span>
                    </div>
                  </div>

                  <div className="text-xs text-zinc-700 leading-relaxed bg-zinc-50 p-3 rounded-lg border border-zinc-200">
                    {transformerStatus.results.smart_diagnostics.advice}
                  </div>

                  <div className="space-y-1 pt-1">
                    <div className="flex justify-between text-[10px] font-mono text-zinc-500">
                      <span>Model Eval Loss: <strong className="text-zinc-800">{transformerStatus.results.smart_diagnostics.eval_loss}</strong></span>
                      <span>Theoretical Random Guess (ln {transformerStatus.results.smart_diagnostics.num_classes}): <strong className="text-zinc-800">{transformerStatus.results.smart_diagnostics.random_guess_loss}</strong></span>
                    </div>
                    <div className="w-full bg-zinc-100 rounded-full h-2 overflow-hidden border border-zinc-200">
                      <div 
                        className={`h-full rounded-full transition-all duration-500 ${
                          transformerStatus.results.smart_diagnostics.phase_color === 'emerald' ? 'bg-emerald-500' : 'bg-blue-600'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(5, transformerStatus.results.smart_diagnostics.loss_reduction_pct))}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Live Terminal Output */}
              <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 shadow-xs">
                <div className="flex items-center justify-between pb-2 border-b border-zinc-800 mb-3">
                  <div className="flex items-center space-x-2 text-zinc-400 text-xs font-mono">
                    <Terminal size={13} />
                    <span>Transformer GPU Live Log Stream</span>
                  </div>
                  <button
                    type="button"
                    onClick={fetchTransformerStatus}
                    className="text-zinc-500 hover:text-zinc-300 transition"
                    title="Refresh Log"
                  >
                    <RefreshCw size={12} />
                  </button>
                </div>
                <div className="font-mono text-xs text-zinc-300 space-y-1 max-h-72 overflow-y-auto scrollbar-thin">
                  {transformerStatus?.recent_logs && transformerStatus.recent_logs.length > 0 ? (
                    transformerStatus.recent_logs.map((line: string, idx: number) => (
                      <div key={idx} className="whitespace-pre-wrap leading-relaxed">{line}</div>
                    ))
                  ) : (
                    <div className="text-zinc-600 py-6 text-center">
                      Siap dijalankan di PC Dosen (RTX 3070). Klik "Jalankan Fine-Tuning GPU" untuk memulai.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 5: Submissions & Ensemble Blending */}
      {activeNlpTab === 'submission' && (
        <div className="space-y-6">
          {/* Header Card */}
          <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-[10px] font-mono uppercase bg-zinc-900 text-white px-2 py-0.5 rounded font-bold">
                  Ensemble & Late-Stage Blending
                </span>
                <span className="text-[10px] font-mono bg-zinc-100 text-zinc-700 px-2 py-0.5 rounded border border-zinc-200 uppercase font-bold">
                  Lexical + Semantic Fusion
                </span>
              </div>
              <h3 className="text-base font-bold text-zinc-950 mt-1">Multi-Model Probability Blending</h3>
              <p className="text-xs text-zinc-500 mt-0.5">
                Gabungkan ketajaman kata kunci TF-IDF (Tab 3) dengan pemahaman konteks IndoBERT (Tab 4) untuk mencetak skor tertinggi.
              </p>
            </div>

            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={() => handleRunEnsemble(true)}
                disabled={isBlending}
                className="flex items-center space-x-1.5 bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition shadow-xs disabled:opacity-50"
              >
                {isBlending ? <Loader2 size={13} className="animate-spin" /> : <TrendingUp size={13} />}
                <span>Auto-Find Optimal Blend</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Left: Interactive Weight Controller */}
            <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-zinc-100">
                <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-1.5">
                  <Sliders size={13} />
                  <span>Blend Ratio Controller</span>
                </h4>
                <span className="text-[10px] font-mono text-zinc-500">Softmax Fusion</span>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs font-mono mb-1.5">
                    <span className="text-zinc-600">TF-IDF (Tab 3):</span>
                    <strong className="text-zinc-900">{ensembleBaselineWeight}%</strong>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="5"
                    value={ensembleBaselineWeight}
                    onChange={e => setEnsembleBaselineWeight(Number(e.target.value))}
                    className="w-full accent-zinc-900 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-zinc-400 mt-1">
                    <span>0% (Pure BERT)</span>
                    <span>50:50</span>
                    <span>100% (Pure TF-IDF)</span>
                  </div>
                </div>

                <div className="p-3 bg-zinc-50 rounded-lg border border-zinc-200 space-y-1.5 text-xs font-mono">
                  <div className="flex justify-between text-zinc-600">
                    <span>TF-IDF Weight (w1):</span>
                    <strong className="text-zinc-900">{(ensembleBaselineWeight / 100).toFixed(2)}</strong>
                  </div>
                  <div className="flex justify-between text-zinc-600">
                    <span>Transformer Weight (w2):</span>
                    <strong className="text-zinc-900">{((100 - ensembleBaselineWeight) / 100).toFixed(2)}</strong>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => handleRunEnsemble(false)}
                  disabled={isBlending}
                  className="w-full flex items-center justify-center space-x-1.5 border border-zinc-300 hover:border-zinc-950 bg-white hover:bg-zinc-50 text-zinc-900 text-xs font-semibold py-2 rounded-lg transition disabled:opacity-50"
                >
                  <Layers size={13} />
                  <span>Apply Custom Blend ({ensembleBaselineWeight}:{100 - ensembleBaselineWeight})</span>
                </button>

                {ensembleError && (
                  <div className="p-3 bg-red-50 border border-red-200 text-red-800 rounded-lg text-xs">
                    {ensembleError}
                  </div>
                )}
              </div>
            </div>

            {/* Right: Ensemble Leaderboard & Metrics */}
            <div className="md:col-span-2 space-y-4">
              {ensembleResult ? (
                <div className="space-y-4">
                  {/* Highlight Cards */}
                  <div className="grid grid-cols-3 gap-3">
                    <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs">
                      <div className="text-[10px] font-mono uppercase text-zinc-400">Baseline (TF-IDF)</div>
                      <div className="text-lg font-bold text-zinc-900 mt-0.5">
                        {(ensembleResult.baseline_macro_f1 * 100).toFixed(2)}%
                      </div>
                      <div className="text-[10px] font-mono text-zinc-500 mt-1">
                        Acc: {(ensembleResult.baseline_accuracy * 100).toFixed(1)}%
                      </div>
                    </div>

                    <div className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs">
                      <div className="text-[10px] font-mono uppercase text-zinc-400">Transformer (BERT)</div>
                      <div className="text-lg font-bold text-zinc-900 mt-0.5">
                        {(ensembleResult.transformer_macro_f1 * 100).toFixed(2)}%
                      </div>
                      <div className="text-[10px] font-mono text-zinc-500 mt-1">
                        Acc: {(ensembleResult.transformer_accuracy * 100).toFixed(1)}%
                      </div>
                    </div>

                    <div className="bg-emerald-50 border border-emerald-300 rounded-xl p-4 shadow-xs">
                      <div className="text-[10px] font-mono uppercase text-emerald-800 font-bold">Ensemble Champion</div>
                      <div className="text-xl font-bold text-emerald-950 mt-0.5">
                        {(ensembleResult.ensemble_macro_f1 * 100).toFixed(2)}%
                      </div>
                      <div className="text-[10px] font-mono text-emerald-700 font-semibold mt-1">
                        +{ensembleResult.f1_gain_pct >= 0 ? ensembleResult.f1_gain_pct : 0.0}% Gain vs Single Best
                      </div>
                    </div>
                  </div>

                  {/* Submission Download Section */}
                  <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs flex items-center justify-between">
                    <div>
                      <div className="text-xs font-bold text-zinc-900">Ensemble Submissions Ready</div>
                      <p className="text-[11px] text-zinc-500 mt-0.5">
                        Bobot optimal yang ditemukan: {Math.round(ensembleResult.optimal_baseline_weight * 100)}% TF-IDF + {Math.round(ensembleResult.optimal_transformer_weight * 100)}% Transformer.
                      </p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <a
                        href={`/api/cases/${selectedCaseId}/download-ensemble-submission`}
                        className="flex items-center space-x-1.5 bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition shadow-xs"
                      >
                        <Download size={13} />
                        <span>Download submission_ensemble.csv</span>
                      </a>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-white border border-zinc-200/80 rounded-xl p-10 text-center shadow-xs space-y-3">
                  <Layers size={28} className="mx-auto text-zinc-400" />
                  <div className="text-sm font-bold text-zinc-900">Siap Melakukan Blending Probabilitas</div>
                  <p className="text-xs text-zinc-500 max-w-md mx-auto">
                    Pastikan Anda telah menjalankan Tab 3 (Baseline) dan Tab 4 (Transformer). Klik tombol "Auto-Find Optimal Blend" di atas untuk mencari kombinasi bobot juara.
                  </p>
                  <button
                    type="button"
                    onClick={() => handleRunEnsemble(true)}
                    disabled={isBlending}
                    className="inline-flex items-center space-x-1.5 bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-medium px-4 py-2 rounded-lg transition"
                  >
                    {isBlending ? <Loader2 size={13} className="animate-spin" /> : <TrendingUp size={13} />}
                    <span>Hitung Optimal Blend Sekarang</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
