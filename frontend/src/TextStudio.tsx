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
  Check
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

              {/* Top Unigrams & Bigrams */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                  <h3 className="text-sm font-bold text-zinc-900 mb-3">Top 20 Most Frequent Words</h3>
                  <div className="space-y-1.5">
                    {edaData.top_words?.map((w: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between text-xs py-1 border-b border-zinc-100 last:border-0">
                        <span className="font-mono text-zinc-800">{w.word}</span>
                        <span className="font-mono text-zinc-400">{w.count?.toLocaleString()}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs">
                  <h3 className="text-sm font-bold text-zinc-900 mb-3">Top 15 Most Frequent Bigrams</h3>
                  <div className="space-y-1.5">
                    {edaData.top_bigrams?.map((b: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between text-xs py-1 border-b border-zinc-100 last:border-0">
                        <span className="font-mono text-zinc-800">{b.bigram}</span>
                        <span className="font-mono text-zinc-400">{b.count?.toLocaleString()}</span>
                      </div>
                    ))}
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
        <div className="bg-white border border-zinc-200/80 rounded-xl p-6 shadow-xs space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-zinc-950">Deep Learning Pipeline (Transformers)</h3>
              <p className="text-xs text-zinc-500 mt-0.5">
                Fine-tune sequence classification models locally from pre-downloaded checkpoints.
              </p>
            </div>
            <span className="text-xs font-mono bg-zinc-100 text-zinc-800 px-3 py-1 rounded-md border border-zinc-200">
              PyTorch / Transformers
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3.5 bg-zinc-50 border border-zinc-200 rounded-lg text-xs">
              <div className="font-bold text-zinc-900 mb-1">Batch Size: 16 (or 32 with grad accum)</div>
              <div className="text-zinc-500">Balances GPU memory utilization and gradient stability.</div>
            </div>
            <div className="p-3.5 bg-zinc-50 border border-zinc-200 rounded-lg text-xs">
              <div className="font-bold text-zinc-900 mb-1">Sequence Length: 256 / 512</div>
              <div className="text-zinc-500">Truncate long articles; keep title + keywords + head/tail of body.</div>
            </div>
            <div className="p-3.5 bg-zinc-50 border border-zinc-200 rounded-lg text-xs">
              <div className="font-bold text-zinc-900 mb-1">Optimizer: AdamW (lr=2e-5)</div>
              <div className="text-zinc-500">With linear warmup and weight decay 0.01.</div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-zinc-900">Standalone Fine-Tuning Script:</span>
              <span className="text-[10px] font-mono text-zinc-400">train_transformer_offline.py</span>
            </div>
            <pre className="p-4 bg-zinc-900 text-zinc-100 rounded-xl text-xs font-mono overflow-x-auto max-h-96">
{`import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 1. Load Cleaned Dataset
train_df = pd.read_csv("cases/${selectedCaseId}/train.csv")
MODEL_PATH = "./models/pretrained-checkpoint" # Local folder path

# 2. Label Encoding
labels = sorted(train_df["${baselineLabelCol || caseDetail?.meta?.target_column || 'category'}"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}
train_df["label"] = train_df["${baselineLabelCol || caseDetail?.meta?.target_column || 'category'}"].map(label2id)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize_fn(examples):
    return tokenizer(examples["${baselineTextCol || 'clean_text'}"], truncation=True, max_length=256, padding="max_length")

# 3. Training Arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro"
)

# 4. Initialize Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    num_labels=len(labels), 
    id2label=id2label, 
    label2id=label2id
)

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "f1_macro": f1_score(y_true, preds, average="macro")
    }`}
            </pre>
          </div>
        </div>
      )}

      {/* Tab 5: Submissions & Inference */}
      {activeNlpTab === 'submission' && (
        <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-zinc-900">Submission & Test Predictions</h3>
              <p className="text-xs text-zinc-500 mt-0.5">
                Inspect generated test predictions and download the final submission file.
              </p>
            </div>
            <a
              href={`/api/cases/${selectedCaseId}/download-submission`}
              className="flex items-center space-x-1.5 bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-medium px-4 py-2 rounded-lg transition"
            >
              <Download size={13} />
              <span>Download submission.csv</span>
            </a>
          </div>

          {baselineResult?.test_predictions_sample ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-zinc-200 text-zinc-400 font-mono text-[10px] uppercase">
                    <th className="pb-2">Test ID</th>
                    <th className="pb-2">Predicted Label</th>
                    <th className="pb-2">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100">
                  {baselineResult.test_predictions_sample.map((row: any, i: number) => (
                    <tr key={i}>
                      <td className="py-2.5 font-mono font-medium text-zinc-900">{row.id}</td>
                      <td className="py-2.5 font-medium text-zinc-800">
                        <span className="px-2 py-0.5 rounded bg-zinc-100 text-zinc-800 border border-zinc-200 font-mono text-[11px]">
                          {row.predicted_label}
                        </span>
                      </td>
                      <td className="py-2.5 font-mono text-zinc-500">{(row.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-8 text-zinc-400 text-xs">
              No test predictions generated yet. Run a baseline model in the "TF-IDF & Baseline CV" tab first.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
