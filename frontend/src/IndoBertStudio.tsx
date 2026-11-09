import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Play, 
  Loader2, 
  Sparkles, 
  Layers, 
  Zap, 
  RotateCcw,
  BarChart3,
  ShieldCheck,
  Send,
  Database
} from 'lucide-react';

interface IndoBertStudioProps {
  selectedCaseId: string;
  caseDetail: any;
  onRefreshCase?: () => void;
}

export const IndoBertStudio: React.FC<IndoBertStudioProps> = ({
  selectedCaseId,
  caseDetail
}) => {
  // GPU Hardware State
  const [gpuStatus, setGpuStatus] = useState<any>(null);
  const [checkingGpu, setCheckingGpu] = useState<boolean>(false);

  // Configuration State
  const [textCols, setTextCols] = useState<string[]>([]);
  const [targetCol, setTargetCol] = useState<string>('');
  const [useTfidf, setUseTfidf] = useState<boolean>(false);
  const [useSentimentFeatures, setUseSentimentFeatures] = useState<boolean>(false);
  const [analysisType, setAnalysisType] = useState<'topic' | 'sentiment'>('topic');
  const [topicModelName, setTopicModelName] = useState('indobenchmark/indobert-base-p1');
  const [sentimentModelName, setSentimentModelName] = useState('AzrilFahmiardi/sdd-sentiment-general');
  const [modelSource, setModelSource] = useState<'online' | 'offline'>('online');
  const [modelName] = useState<string>('indobenchmark/indobert-base-p1');
  const [maxSeqLength, setMaxSeqLength] = useState<number>(256);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [pooling, setPooling] = useState<'cls' | 'mean'>('cls');
  const [useGpu, setUseGpu] = useState<boolean>(true);
  const [nSplits, setNSplits] = useState<number>(5);
  const [tfidfMaxFeatures, setTfidfMaxFeatures] = useState<number>(15000);
  const [tfidfNgramMax, setTfidfNgramMax] = useState<number>(2);

  // Classifier Selection
  const [useLgbm, setUseLgbm] = useState<boolean>(true);
  const [useCatboost, setUseCatboost] = useState<boolean>(true);
  const [useXgb, setUseXgb] = useState<boolean>(true);

  // Training & Execution State
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainState, setTrainState] = useState<any>(null);
  const [results, setResults] = useState<any>(null);

  // Interactive Live Inference Sandbox State
  const [sandboxTexts, setSandboxTexts] = useState<Record<string, string>>({});
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [testFiles, setTestFiles] = useState<Array<{ path: string; filename: string; size_mb: number }>>([]);
  const [selectedTestFile, setSelectedTestFile] = useState<string>('');
  const [isExportingTest, setIsExportingTest] = useState<boolean>(false);

  const availableCols: string[] = caseDetail?.dataset_columns || [];
  const sandboxCols: string[] = results?.meta?.text_cols || textCols;
  const featureLabel = useTfidf ? 'IndoBERT + TF-IDF' : 'IndoBERT';
  const resultEncoderLabel = results?.meta?.model_name && results.meta.model_name !== 'TF-IDF'
    ? String(results.meta.model_name).split('/').pop()
    : 'IndoBERT';
  const resultFeatureLabel = results?.meta?.feature_extractor === 'both'
    ? `${resultEncoderLabel} + TF-IDF`
    : (results?.meta?.feature_extractor === 'tfidf' ? 'TF-IDF' : (results?.meta?.feature_extractor === 'pretrained_sentiment' ? 'Indonesian Sentiment Model' : resultEncoderLabel));
  const isSentiment = analysisType === 'sentiment';

  // Fetch GPU Status
  const fetchGpuStatus = async () => {
    setCheckingGpu(true);
    try {
      const res = await fetch('/api/system/gpu-status');
      if (res.ok) {
        const data = await res.json();
        setGpuStatus(data);
        if (data.cuda_available) {
          setUseGpu(true);
        }
      }
    } catch (e) {
      console.error('Failed to fetch GPU status:', e);
    } finally {
      setCheckingGpu(false);
    }
  };

  // Fetch current IndoBERT state and existing results
  const fetchIndoBertState = async () => {
    if (!selectedCaseId) return;
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/indobert/state`);
      if (res.ok) {
        const data = await res.json();
        setTrainState(data);
        if (data.results) {
          setResults(data.results);
          if (data.results.meta?.sentiment_model_name) {
            setSentimentModelName(data.results.meta.sentiment_model_name);
          }
          if (data.results.meta?.model_source === 'online' || data.results.meta?.model_source === 'offline') {
            setModelSource(data.results.meta.model_source);
          }
          if (data.results.meta?.model_name && ['indobenchmark/indobert-base-p1', 'indolem/indobertweet-base-uncased', 'flax-community/indonesian-roberta-base'].includes(data.results.meta.model_name)) {
            setTopicModelName(data.results.meta.model_name);
          }
        }
        if (data.status === 'running') {
          setIsTraining(true);
        } else {
          setIsTraining(false);
        }
      }
    } catch (e) {
      console.error('Failed to fetch IndoBERT state:', e);
    }
  };

  useEffect(() => {
    fetchGpuStatus();
  }, []);

  useEffect(() => {
    if (caseDetail) {
      const meta = caseDetail.meta || {};
      const activeTextCol = meta.text_column || (availableCols.includes('clean_text') ? 'clean_text' : (availableCols.includes('content') ? 'content' : (availableCols.includes('text') ? 'text' : availableCols[0] || '')));
      setTextCols(activeTextCol ? [activeTextCol] : []);
      setTargetCol(meta.target_column || meta.target_col || (availableCols.includes('category') ? 'category' : (availableCols.includes('label') ? 'label' : '')));
    }
    fetchIndoBertState();
  }, [selectedCaseId, caseDetail]);

  // Polling during training
  useEffect(() => {
    let interval: any = null;
    if (isTraining) {
      interval = setInterval(() => {
        fetchIndoBertState();
      }, 2000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isTraining]);

  // Seed one sandbox input per trained text column
  useEffect(() => {
    const cols: string[] = results?.meta?.text_cols || [];
    if (cols.length === 0) return;
    setSandboxTexts((prev) => {
      const next = { ...prev };
      cols.forEach((c, i) => {
        if (next[c] === undefined) {
          next[c] = i === 0
            ? 'Presiden meluncurkan program transformasi kecerdasan buatan nasional untuk mempercepat adopsi teknologi di sektor industri dan pendidikan digital Indonesia.'
            : '';
        }
      });
      return next;
    });
  }, [results]);

  useEffect(() => {
    if (results?.meta?.feature_extractor !== 'pretrained_sentiment') return;
    fetch('/api/scan-workspace-csvs')
      .then((res) => res.ok ? res.json() : Promise.reject(new Error('Tidak dapat memuat daftar file CSV')))
      .then((data) => {
        const files = (data.files || []).filter((file: any) => ['CSV', 'XLSX'].includes(file.format));
        setTestFiles(files);
        setSelectedTestFile((current) => files.some((file: any) => file.path === current) ? current : '');
      })
      .catch((err) => console.error('Gagal memuat file data tes:', err));
  }, [selectedCaseId, results?.meta?.feature_extractor]);

  // Trigger Training
  const handleStartTraining = async () => {
    if (!selectedCaseId) return;
    const selectedModels: string[] = [];
    if (useLgbm) selectedModels.push('lgbm');
    if (useCatboost) selectedModels.push('catboost');
    if (useXgb) selectedModels.push('xgb');

    if (textCols.length === 0) {
      alert('Pilih setidaknya satu kolom teks sebagai fitur!');
      return;
    }
    if (!isSentiment && selectedModels.length === 0) {
      alert('Pilih setidaknya satu algoritma classifier (LightGBM, CatBoost, atau XGBoost)!');
      return;
    }

    setIsTraining(true);
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/indobert/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          text_cols: textCols,
          target_col: targetCol,
          feature_extractor: useTfidf ? 'both' : 'indobert',
          text_task: analysisType === 'sentiment' ? 'sentiment_multiclass' : 'topic_classification',
          sentiment_model_name: sentimentModelName,
          model_source: modelSource,
          feature_model_name: topicModelName,
          use_sentiment_features: useSentimentFeatures,
          tfidf_max_features: tfidfMaxFeatures,
          tfidf_ngram_min: 1,
          tfidf_ngram_max: tfidfNgramMax,
          models: selectedModels,
          n_splits: nSplits,
          batch_size: batchSize,
          max_length: maxSeqLength,
          pooling: pooling,
          use_gpu: useGpu
        })
      });

      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || 'Gagal memulai Language Studio');
        setIsTraining(false);
      } else {
        fetchIndoBertState();
      }
    } catch (e: any) {
      alert('Network error: ' + e.message);
      setIsTraining(false);
    }
  };

  // Test Interactive Inference
  const handleTestPrediction = async () => {
    if (!Object.values(sandboxTexts).some((t) => t.trim())) return;
    setIsPredicting(true);
    try {
      const res = await fetch(`/api/cases/${selectedCaseId}/indobert/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: sandboxTexts })
      });
      if (res.ok) {
        const data = await res.json();
        setPredictionResult(data);
      } else {
        const err = await res.json();
        alert(err.detail || 'Prediction failed');
      }
    } catch (e: any) {
      alert('Error predicting: ' + e.message);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleLabelTestFile = async () => {
    if (!selectedTestFile) return;
    setIsExportingTest(true);
    try {
      const response = await fetch(`/api/cases/${selectedCaseId}/indobert/predict-file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: selectedTestFile, text_cols: results?.meta?.text_cols || textCols })
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Gagal memprediksi file tes');
      }
      const blob = await response.blob();
      const disposition = response.headers.get('Content-Disposition') || '';
      const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
      const filename = encodedName ? decodeURIComponent(encodedName) : 'sentiment_labeled_test.csv';
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (error: any) {
      alert(error.message || 'Gagal memprediksi file tes');
    } finally {
      setIsExportingTest(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Studio Header & Hardware Acceleration Card */}
      <div className="bg-white border border-zinc-200/80 rounded-2xl p-6 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2.5">
            <span className="p-2 bg-emerald-50 text-emerald-600 rounded-xl border border-emerald-200/60">
              <Sparkles size={20} />
            </span>
            <div>
              <h2 className="text-xl font-bold text-zinc-950 flex items-center gap-2">
                Language Studio
                <span className="text-xs px-2.5 py-0.5 rounded-full font-mono font-medium bg-emerald-100 text-emerald-800 border border-emerald-200">
                  indobenchmark/indobert-base-p1
                </span>
              </h2>
              <p className="text-xs text-zinc-500 mt-0.5">
                {isSentiment ? 'Analisis sentimen bahasa Indonesia dengan IndoBERT yang sudah di-fine-tune untuk klasifikasi sentimen.' : 'Indonesian text feature extraction with IndoBERT embeddings plus optional TF-IDF features, with downstream GBDT classification (LightGBM, CatBoost &amp; XGBoost).'}
              </p>
            </div>
          </div>
        </div>

        {/* Live GPU Hardware Status Card */}
        <div className="flex items-center gap-3 bg-zinc-50 border border-zinc-200 rounded-xl px-4 py-2.5">
          <div className="flex items-center space-x-2">
            <Cpu className={gpuStatus?.cuda_available ? 'text-emerald-500 animate-pulse' : 'text-zinc-400'} size={18} />
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-semibold text-zinc-900">
                  {gpuStatus?.cuda_available ? gpuStatus.device_name : 'CPU Execution Mode'}
                </span>
                {gpuStatus?.cuda_available && (
                  <span className="px-1.5 py-0.2 text-[10px] font-bold bg-emerald-500 text-white rounded">
                    CUDA ON
                  </span>
                )}
              </div>
              <div className="text-[11px] text-zinc-500 font-mono">
                {gpuStatus?.cuda_available
                  ? `VRAM: ${gpuStatus.free_memory_mb} MB / ${gpuStatus.total_memory_mb} MB free`
                  : 'Hardware: Standard CPU (Multi-threaded)'}
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={fetchGpuStatus}
            disabled={checkingGpu}
            title="Refresh GPU hardware check"
            className="p-1.5 text-zinc-400 hover:text-zinc-700 hover:bg-zinc-200/60 rounded-md transition-all ml-1"
          >
            <RotateCcw size={14} className={checkingGpu ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Visual Architecture Banner */}
      <div className="bg-gradient-to-r from-zinc-900 via-zinc-800 to-zinc-900 text-zinc-100 p-4 rounded-xl border border-zinc-800 shadow-xs flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center space-x-2 font-medium">
          <Layers size={16} className="text-emerald-400" />
          <span>Arsitektur Pipeline:</span>
        </div>
        <div className="flex items-center space-x-2 font-mono text-[11px]">
          <span className="bg-zinc-800 px-2.5 py-1 rounded border border-zinc-700 text-zinc-300">
            1. Raw Text ({textCols.length > 0 ? textCols.join(' + ') : 'content'})
          </span>
          <span className="text-zinc-500">&rarr;</span>
          <span className="bg-emerald-950/80 text-emerald-300 px-2.5 py-1 rounded border border-emerald-800/80 font-semibold">
            2. {isSentiment ? 'IndoBERT Sentiment Classifier' : `${topicModelName.split('/').pop()}${useTfidf ? ' + TF-IDF' : ' Embeddings'}`}
          </span>
          <span className="text-zinc-500">&rarr;</span>
          <span className="bg-blue-950/80 text-blue-300 px-2.5 py-1 rounded border border-blue-800/80">
            {isSentiment ? '3. Pretrained Sentiment Classifier' : '3. LightGBM + CatBoost + XGBoost'}
          </span>
          <span className="text-zinc-500">&rarr;</span>
          <span className="bg-amber-950/80 text-amber-300 px-2.5 py-1 rounded border border-amber-800/80 font-bold">
            {isSentiment ? '4. Negative / Neutral / Positive' : '4. Soft Voting Ensemble'}
          </span>
        </div>
      </div>

      {/* Control Panel: Configuration & Parameters */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Input Columns & Transformers Parameters */}
        <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs space-y-4">
          <h3 className="text-sm font-bold text-zinc-900 flex items-center gap-1.5">
            <Database size={15} className="text-zinc-500" />
            1. Kolom &amp; Feature Extractor
          </h3>

          <div>
            <label className="block text-xs font-semibold text-zinc-700 mb-1">Jenis Analisis</label>
            <select value={analysisType} onChange={(e) => setAnalysisType(e.target.value as 'topic' | 'sentiment')} className="w-full text-xs bg-zinc-50 border border-zinc-200 rounded-lg p-2">
              <option value="topic">Topik</option><option value="sentiment">Sentimen (multikelas)</option>
            </select>
          </div>

          <div className={`p-3 rounded-lg border ${modelSource === 'offline' ? 'border-amber-200 bg-amber-50/50' : 'border-sky-200 bg-sky-50/50'}`}>
            <label className="block text-xs font-semibold text-zinc-700 mb-1.5">Sumber Model</label>
            <select value={modelSource} onChange={(e) => setModelSource(e.target.value as 'online' | 'offline')} className="w-full text-xs bg-white border border-zinc-200 rounded-lg p-2">
              <option value="online">Online — Hugging Face Hub</option>
              <option value="offline">Offline — checkpoint lokal</option>
            </select>
            <p className="text-[11px] text-zinc-500 mt-1.5">
              {modelSource === 'offline'
                ? 'Mode offline hanya membuka model dari folder models/ atau models-bert/; model yang belum tersedia tidak akan diunduh.'
                : 'Mode online memuat model dari Hugging Face dan dapat mengunduhnya saat pertama kali digunakan.'}
            </p>
          </div>

          {isSentiment && (
            <div className="p-3 rounded-lg border border-violet-200 bg-violet-50/50">
              <label className="block text-xs font-semibold text-zinc-700 mb-1.5">Model Sentimen / Feature Extractor</label>
              <select value={sentimentModelName} onChange={(e) => setSentimentModelName(e.target.value)} className="w-full text-xs bg-white border border-zinc-200 rounded-lg p-2 font-mono">
                <option value="AzrilFahmiardi/sdd-sentiment-general">IndoBERT Sentiment — Berita Indonesia</option>
                <option value="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis">IndoBERTweet — Sentiment</option>
                <option value="w11wo/indonesian-roberta-base-sentiment-classifier">Indonesian RoBERTa — Sentiment</option>
              </select>
              <p className="text-[11px] text-zinc-500 mt-1.5">Model terpilih akan mengklasifikasikan sentimen secara langsung dan dipakai juga untuk pelabelan file tes.</p>
            </div>
          )}

          {!isSentiment && (
            <div className="p-3 rounded-lg border border-sky-200 bg-sky-50/50">
              <label className="block text-xs font-semibold text-zinc-700 mb-1.5">Model Feature Extractor Topik</label>
              <select value={topicModelName} onChange={(e) => setTopicModelName(e.target.value)} className="w-full text-xs bg-white border border-zinc-200 rounded-lg p-2 font-mono">
                <option value="indobenchmark/indobert-base-p1">IndoBERT Base P1</option>
                <option value="indolem/indobertweet-base-uncased">IndoBERTweet Base</option>
                <option value="flax-community/indonesian-roberta-base">Indonesian RoBERTa Base</option>
              </select>
              <p className="text-[11px] text-zinc-500 mt-1.5">Embedding dari model ini akan dipakai oleh classifier GBDT untuk klasifikasi topik.</p>
            </div>
          )}

          <div>
            <label className="block text-xs font-semibold text-zinc-700 mb-1">
              Text Feature Columns * <span className="font-normal text-zinc-400">(bisa lebih dari satu)</span>
            </label>
            <div className="max-h-40 overflow-y-auto border border-zinc-200 rounded-lg bg-zinc-50 p-2 space-y-1">
              {availableCols.map((c) => (
                <label key={c} className="flex items-center space-x-2 text-xs cursor-pointer">
                  <input
                    type="checkbox"
                    checked={textCols.includes(c)}
                    onChange={(e) => {
                      setTextCols((prev) =>
                        e.target.checked ? [...prev, c] : prev.filter((x) => x !== c)
                      );
                    }}
                    className="rounded border-zinc-300 text-emerald-600 focus:ring-emerald-500"
                  />
                  <span className="font-mono text-zinc-800">{c}</span>
                </label>
              ))}
            </div>
          </div>

          <div className={`p-3 rounded-lg border transition-all ${useTfidf ? 'bg-emerald-50/50 border-emerald-200' : 'bg-zinc-50 border-zinc-200'} ${isSentiment ? 'hidden' : ''}`}>
            <label className="flex items-start space-x-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={useTfidf}
                onChange={(e) => setUseTfidf(e.target.checked)}
                className="mt-0.5 rounded border-zinc-300 text-emerald-600 focus:ring-emerald-500"
              />
              <div>
                <div className="text-xs font-bold text-zinc-900">Tambahkan Fitur TF-IDF</div>
                <div className="text-[11px] text-zinc-500 mt-0.5">
                  TF-IDF digabung dengan embedding IndoBERT sebelum masuk ke GBDT. Tanpa centang, hanya IndoBERT yang dipakai.
                </div>
              </div>
            </label>
          </div>

          <label className={`flex items-start gap-2.5 p-3 rounded-lg border cursor-pointer ${useSentimentFeatures ? 'bg-amber-50 border-amber-200' : 'bg-zinc-50 border-zinc-200'} ${isSentiment ? 'hidden' : ''}`}>
            <input type="checkbox" checked={useSentimentFeatures} onChange={(e) => setUseSentimentFeatures(e.target.checked)} className="mt-0.5 rounded border-zinc-300 text-emerald-600" />
            <span><span className="block text-xs font-bold text-zinc-900">Gunakan fitur sentiment (lexicon + negasi + emoji)</span><span className="text-[11px] text-zinc-500">Memperkuat sinyal polaritas per kolom teks.</span></span>
          </label>

          <div>
            <label className="block text-xs font-semibold text-zinc-700 mb-1">
              {isSentiment ? 'Kolom Label (opsional untuk evaluasi)' : 'Target Classification Column *'}
            </label>
            <select
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="w-full text-xs bg-zinc-50 border border-zinc-200 rounded-lg p-2 font-mono focus:outline-none focus:ring-1 focus:ring-zinc-900"
            >
              {availableCols.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2">
            <div>
              <label className="block text-xs font-semibold text-zinc-700 mb-1">
                Max Seq Length
              </label>
              <select
                value={maxSeqLength}
                onChange={(e) => setMaxSeqLength(Number(e.target.value))}
                className="w-full text-xs bg-zinc-50 border border-zinc-200 rounded-lg p-2 font-mono"
              >
                <option value={128}>128 tokens</option>
                <option value={256}>256 tokens (Default)</option>
                <option value={512}>512 tokens (Full)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-zinc-700 mb-1">
                Batch Size
              </label>
              <select
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full text-xs bg-zinc-50 border border-zinc-200 rounded-lg p-2 font-mono"
              >
                <option value={16}>16 (Low VRAM)</option>
                <option value={32}>32 (Optimal)</option>
                <option value={64}>64 (Fast GPU)</option>
              </select>
            </div>
          </div>

          {useTfidf && (
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-zinc-700 mb-1">
                  Max TF-IDF Features / Kolom
                </label>
                <select
                  value={tfidfMaxFeatures}
                  onChange={(e) => setTfidfMaxFeatures(Number(e.target.value))}
                  className="w-full text-xs bg-zinc-50 border border-zinc-200 rounded-lg p-2 font-mono"
                >
                  <option value={5000}>5.000 (Ringan)</option>
                  <option value={15000}>15.000 (Default)</option>
                  <option value={30000}>30.000 (Maksimal)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-zinc-700 mb-1">
                  TF-IDF N-Gram Range
                </label>
                <select
                  value={tfidfNgramMax}
                  onChange={(e) => setTfidfNgramMax(Number(e.target.value))}
                  className="w-full text-xs bg-zinc-50 border border-zinc-200 rounded-lg p-2 font-mono"
                >
                  <option value={1}>Unigram (1,1)</option>
                  <option value={2}>Unigram + Bigram (1,2)</option>
                  <option value={3}>Unigram s/d Trigram (1,3)</option>
                </select>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-zinc-700 mb-1">
                Pooling Strategy
              </label>
              <select
                value={pooling}
                onChange={(e) => setPooling(e.target.value as any)}
                className="w-full text-xs bg-zinc-50 border border-zinc-200 rounded-lg p-2 font-mono"
              >
                <option value="cls">[CLS] Token (Pooler)</option>
                <option value="mean">Mean Pooling</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-zinc-700 mb-1">
                CV Folds
              </label>
              <select
                value={nSplits}
                onChange={(e) => setNSplits(Number(e.target.value))}
                className="w-full text-xs bg-zinc-50 border border-zinc-200 rounded-lg p-2 font-mono"
              >
                <option value={3}>3-Fold Stratified</option>
                <option value={5}>5-Fold Stratified (Recommended)</option>
                <option value={10}>10-Fold Stratified</option>
              </select>
            </div>
          </div>

          <div className="pt-2">
            <label className="flex items-center space-x-2 text-xs font-medium text-zinc-800 cursor-pointer">
              <input
                type="checkbox"
                checked={useGpu}
                onChange={(e) => setUseGpu(e.target.checked)}
                className="rounded border-zinc-300 text-emerald-600 focus:ring-emerald-500"
              />
              <span>Gunakan GPU Acceleration jika tersedia</span>
            </label>
          </div>
        </div>

        {/* Center: Downstream Machine Learning Classifiers */}
        <div className={`bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs space-y-4 ${isSentiment ? 'hidden' : ''}`}>
          <h3 className="text-sm font-bold text-zinc-900 flex items-center gap-1.5">
            <BarChart3 size={15} className="text-zinc-500" />
            2. Downstream ML Classifiers
          </h3>
          <p className="text-xs text-zinc-500">
            Model GBDT yang dilatih di depan fitur teks yang diekstrak:
          </p>

          <div className="space-y-3">
            {/* LightGBM */}
            <div className={`p-3 rounded-lg border transition-all ${useLgbm ? 'bg-emerald-50/50 border-emerald-200' : 'bg-zinc-50 border-zinc-200 opacity-60'}`}>
              <label className="flex items-start space-x-2.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useLgbm}
                  onChange={(e) => setUseLgbm(e.target.checked)}
                  className="mt-0.5 rounded border-zinc-300 text-emerald-600 focus:ring-emerald-500"
                />
                <div>
                  <div className="text-xs font-bold text-zinc-900">LightGBM Classifier</div>
                  <div className="text-[11px] text-zinc-500 mt-0.5">
                    Leaf-wise tree growth, ultra-cepat untuk fitur dense multi-dimensi.
                  </div>
                </div>
              </label>
            </div>

            {/* CatBoost */}
            <div className={`p-3 rounded-lg border transition-all ${useCatboost ? 'bg-amber-50/50 border-amber-200' : 'bg-zinc-50 border-zinc-200 opacity-60'}`}>
              <label className="flex items-start space-x-2.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useCatboost}
                  onChange={(e) => setUseCatboost(e.target.checked)}
                  className="mt-0.5 rounded border-zinc-300 text-amber-600 focus:ring-amber-500"
                />
                <div>
                  <div className="text-xs font-bold text-zinc-900">CatBoost Classifier</div>
                  <div className="text-[11px] text-zinc-500 mt-0.5">
                    Symmetric trees dengan regularisasi kuat, kebal terhadap noise &amp; overfitting.
                  </div>
                </div>
              </label>
            </div>

            {/* XGBoost */}
            <div className={`p-3 rounded-lg border transition-all ${useXgb ? 'bg-blue-50/50 border-blue-200' : 'bg-zinc-50 border-zinc-200 opacity-60'}`}>
              <label className="flex items-start space-x-2.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useXgb}
                  onChange={(e) => setUseXgb(e.target.checked)}
                  className="mt-0.5 rounded border-zinc-300 text-blue-600 focus:ring-blue-500"
                />
                <div>
                  <div className="text-xs font-bold text-zinc-900">XGBoost Classifier</div>
                  <div className="text-[11px] text-zinc-500 mt-0.5">
                    Hist gradient boosting dengan akselerasi GPU CUDA.
                  </div>
                </div>
              </label>
            </div>
          </div>

          <div className="p-3 bg-zinc-900 text-zinc-200 rounded-lg text-xs space-y-1">
            <div className="font-semibold text-emerald-400 flex items-center gap-1">
              <Zap size={13} />
              Auto-Ensemble (Soft Voting)
            </div>
            <div className="text-[11px] text-zinc-400">
              Jika memilih &gt; 1 model, sistem otomatis menghitung gabungan probabilitas out-of-fold.
            </div>
          </div>
        </div>

        {/* Right: Trigger & Pipeline Summary */}
        <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-zinc-900 flex items-center gap-1.5">
              <ShieldCheck size={15} className="text-zinc-500" />
              3. Eksekusi Pelatihan
            </h3>
            <p className="text-xs text-zinc-500 leading-relaxed">
              {isSentiment ? `Model sentimen memprediksi negatif, netral, atau positif. Sumber model: ${modelSource}.` : `Fitur teks diekstrak lalu model GBDT dilatih dengan validasi silang. Sumber model: ${modelSource}.`}
            </p>

            <div className="space-y-1.5 text-xs text-zinc-600 border-t border-zinc-100 pt-3">
              <div className="flex justify-between">
                <span>Model BERT:</span>
                <span className="font-mono font-medium text-zinc-900">{isSentiment ? sentimentModelName.split('/').pop() : topicModelName.split('/').pop()}</span>
              </div>
              <div className="flex justify-between">
                <span>Feature Extractor:</span>
                <span className="font-mono font-medium text-zinc-900">
                  {isSentiment ? 'Pretrained sentiment classifier' : featureLabel}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Fitur Teks:</span>
                <span className="font-mono font-medium text-zinc-900 text-right max-w-[60%] truncate" title={textCols.join(', ')}>
                  {textCols.length > 0 ? textCols.join(', ') : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Target:</span>
                <span className="font-mono font-medium text-zinc-900">{targetCol || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span>Akselerasi:</span>
                <span className="font-medium text-emerald-600">
                  {useGpu && gpuStatus?.cuda_available ? 'GPU (CUDA)' : 'CPU Multi-Core'}
                </span>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={handleStartTraining}
            disabled={isTraining || textCols.length === 0 || (!targetCol && !isSentiment)}
            className={`w-full py-3 px-4 rounded-xl text-xs font-bold transition-all flex items-center justify-center space-x-2 shadow-sm ${
              isTraining || textCols.length === 0 || (!targetCol && !isSentiment)
                ? 'bg-zinc-200 text-zinc-400 cursor-not-allowed'
                : 'bg-zinc-950 text-white hover:bg-zinc-800'
            }`}
          >
            {isTraining ? (
              <>
                <Loader2 className="animate-spin" size={16} />
                <span>{isSentiment ? 'Analisis Sentimen Sedang Berjalan...' : 'Pelatihan IndoBERT Sedang Berjalan...'}</span>
              </>
            ) : (
              <>
                <Play size={16} />
                <span>{isSentiment ? 'Mulai Analisis Sentimen' : `Mulai Pelatihan ${featureLabel} + GBDT`}</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Real-Time Training Progress & Terminal Logs */}
      {(isTraining || (trainState?.logs && trainState.logs.length > 0)) && (
        <div className="bg-zinc-950 border border-zinc-800 text-zinc-200 rounded-xl p-5 shadow-lg space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <div className="flex items-center space-x-2">
              <span className={`h-2.5 w-2.5 rounded-full ${isTraining ? 'bg-emerald-400 animate-ping' : 'bg-emerald-500'}`} />
              <span className="text-xs font-mono font-semibold text-zinc-100">
                Live Language Studio Execution Terminal ({trainState?.stage || 'idle'})
              </span>
            </div>
            <span className="text-xs font-mono text-zinc-400">
              Progress: {trainState?.progress || 0}%
            </span>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-emerald-500 h-1.5 transition-all duration-300"
              style={{ width: `${trainState?.progress || 0}%` }}
            />
          </div>

          {/* Console Log Area */}
          <div className="font-mono text-xs bg-zinc-900/80 p-3 rounded-lg max-h-48 overflow-y-auto space-y-1 text-zinc-300">
            {trainState?.logs?.map((l: string, i: number) => (
              <div key={i} className="leading-relaxed">
                {l}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Results Dashboard */}
      {results && (
        <div className="space-y-6">
          {/* Top Metric Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {results.ensemble && (
              <div className="bg-gradient-to-br from-emerald-50 to-teal-50 border border-emerald-200/80 rounded-xl p-4 shadow-xs">
                <div className="text-[11px] font-semibold text-emerald-800 uppercase tracking-wider">
                  Ensemble Accuracy
                </div>
                <div className="text-2xl font-bold text-emerald-950 mt-1">
                  {results.ensemble.accuracy}%
                </div>
                <div className="text-xs text-emerald-700 mt-1 font-medium">
                  Macro F1: {results.ensemble.f1_macro}% (+{results.ensemble.gain_over_best_single}% gain)
                </div>
              </div>
            )}

            {Object.keys(results.models || {}).map((mKey: string) => {
              const m = results.models[mKey];
              return (
                <div key={mKey} className="bg-white border border-zinc-200/80 rounded-xl p-4 shadow-xs">
                  <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
                    {m.name}
                  </div>
                  <div className="text-2xl font-bold text-zinc-900 mt-1">
                    {m.accuracy}%
                  </div>
                  <div className="text-xs text-zinc-600 mt-1">
                    Macro F1: <span className="font-bold">{m.f1_macro}%</span> | {m.training_time_sec}s
                  </div>
                </div>
              );
            })}
          </div>

          {/* Comparison Table */}
          <div className="bg-white border border-zinc-200/80 rounded-xl overflow-hidden shadow-xs">
            <div className="p-4 border-b border-zinc-200 bg-zinc-50/50 flex items-center justify-between">
              <div>
                <h4 className="text-sm font-bold text-zinc-900">Perbandingan Model Klasifikasi ({resultFeatureLabel} Features)</h4>
                <p className="text-xs text-zinc-500 mt-0.5">
                  {results.meta?.feature_extractor === 'pretrained_sentiment' ? 'Hasil model sentimen pretrained' : 'Hasil evaluasi out-of-fold validasi silang'} {results.meta?.total_samples?.toLocaleString()} sampel berita
                </p>
              </div>
              <span className="text-xs font-mono bg-zinc-100 text-zinc-700 px-2.5 py-1 rounded border border-zinc-200">
                {results.classes?.length} Classes: {results.classes?.join(', ')}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-zinc-50 text-zinc-600 font-semibold border-b border-zinc-200">
                  <tr>
                    <th className="py-2.5 px-4">Model Architecture</th>
                    <th className="py-2.5 px-4">Accuracy</th>
                    <th className="py-2.5 px-4">Macro F1</th>
                    <th className="py-2.5 px-4">Weighted F1</th>
                    <th className="py-2.5 px-4">Precision</th>
                    <th className="py-2.5 px-4">Recall</th>
                    <th className="py-2.5 px-4">Training Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-200 font-mono">
                  {results.ensemble && (
                    <tr className="bg-emerald-50/40 font-semibold text-emerald-950">
                      <td className="py-2.5 px-4 font-sans flex items-center gap-1.5">
                        <Sparkles size={14} className="text-emerald-600" />
                        {results.ensemble.name}
                      </td>
                      <td className="py-2.5 px-4 font-bold">{results.ensemble.accuracy}%</td>
                      <td className="py-2.5 px-4 font-bold">{results.ensemble.f1_macro}%</td>
                      <td className="py-2.5 px-4">{results.ensemble.f1_weighted}%</td>
                      <td className="py-2.5 px-4">-</td>
                      <td className="py-2.5 px-4">-</td>
                      <td className="py-2.5 px-4 font-sans text-emerald-700">Fast Blend</td>
                    </tr>
                  )}
                  {Object.keys(results.models || {}).map((mKey: string) => {
                    const m = results.models[mKey];
                    return (
                      <tr key={mKey} className="hover:bg-zinc-50/60">
                        <td className="py-2.5 px-4 font-sans font-medium text-zinc-900">
                          {resultFeatureLabel} + {m.name}
                        </td>
                        <td className="py-2.5 px-4 font-bold text-zinc-900">{m.accuracy}%</td>
                        <td className="py-2.5 px-4 font-bold text-emerald-700">{m.f1_macro}%</td>
                        <td className="py-2.5 px-4 text-zinc-700">{m.f1_weighted}%</td>
                        <td className="py-2.5 px-4 text-zinc-700">{m.precision_macro}%</td>
                        <td className="py-2.5 px-4 text-zinc-700">{m.recall_macro}%</td>
                        <td className="py-2.5 px-4 font-sans text-zinc-500">{m.training_time_sec}s</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {Object.entries(results.models || {}).map(([key, model]: [string, any]) => (
              <div key={`${key}-class-metrics`} className="mt-3 rounded-lg border border-zinc-200 overflow-hidden">
                <div className="px-3 py-2 bg-zinc-50 text-xs font-semibold">Recall &amp; F1 per kelas — {model.name}</div>
                <div className="flex flex-wrap gap-2 p-3">
                  {Object.entries(model.per_class || {}).map(([label, metric]: [string, any]) => (
                    <span key={label} className={`text-[11px] px-2 py-1 rounded border ${label.toLowerCase().includes('neutral') || label.toLowerCase().includes('netral') ? 'bg-amber-50 border-amber-200 text-amber-900' : 'bg-zinc-50 border-zinc-200 text-zinc-700'}`}>
                      {label}: recall {metric.recall}% · F1 {metric.f1_score}%
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {results.meta?.feature_extractor === 'pretrained_sentiment' && (
            <div className="bg-white border border-emerald-200 rounded-xl p-5 shadow-xs space-y-3">
              <div>
                <h4 className="text-sm font-bold text-zinc-900">Labeli File Data Tes &amp; Export CSV</h4>
                <p className="text-xs text-zinc-500 mt-1">Pilih CSV atau XLSX dari workspace. Model sentimen yang sudah aktif akan memberi label pada semua baris dan mengunduh hasilnya sebagai CSV.</p>
              </div>
              <div className="flex flex-col sm:flex-row gap-2">
                <select value={selectedTestFile} onChange={(e) => setSelectedTestFile(e.target.value)} className="flex-1 text-xs bg-zinc-50 border border-zinc-200 rounded-lg p-2 font-mono">
                  <option value="">Pilih file CSV atau XLSX...</option>
                  {testFiles.map((file) => <option key={file.path} value={file.path}>{file.path} ({file.size_mb} MB)</option>)}
                </select>
                <button type="button" onClick={handleLabelTestFile} disabled={!selectedTestFile || isExportingTest} className="px-4 py-2 bg-emerald-700 text-white rounded-lg text-xs font-semibold hover:bg-emerald-800 disabled:opacity-50 flex items-center justify-center gap-2">
                  {isExportingTest ? <><Loader2 size={14} className="animate-spin" /> Memberi label...</> : <><Database size={14} /> Labeli &amp; Unduh CSV</>}
                </button>
              </div>
              <p className="text-[11px] text-zinc-500">Kolom teks yang dipilih saat analisis harus tersedia pada file tes.</p>
            </div>
          )}

          {/* Interactive Live Inference Sandbox */}
          <div className="bg-white border border-zinc-200/80 rounded-xl p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-bold text-zinc-900 flex items-center gap-1.5">
                  <Send size={15} className="text-zinc-500" />
                  Interactive {resultFeatureLabel} Inference Sandbox
                </h4>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Uji coba klasifikasi teks langsung menggunakan stack {resultFeatureLabel} &amp; GBDT yang sudah dilatih.
                </p>
              </div>
            </div>

            <div className="space-y-3">
              {sandboxCols.map((col) => (
                <div key={col}>
                  <label className="block text-xs font-semibold text-zinc-700 mb-1 font-mono">
                    {col}
                  </label>
                  <textarea
                    value={sandboxTexts[col] || ''}
                    onChange={(e) => setSandboxTexts((prev) => ({ ...prev, [col]: e.target.value }))}
                    rows={3}
                    placeholder={`Ketik isi kolom ${col} di sini...`}
                    className="w-full text-xs p-3 bg-zinc-50 border border-zinc-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-zinc-900"
                  />
                </div>
              ))}

              <div className="flex items-center justify-between">
                <button
                  type="button"
                  onClick={handleTestPrediction}
                  disabled={isPredicting || !Object.values(sandboxTexts).some((t) => t.trim())}
                  className="px-4 py-2 bg-zinc-900 text-white rounded-lg text-xs font-semibold hover:bg-zinc-800 transition-all flex items-center space-x-1.5 disabled:opacity-50"
                >
                  {isPredicting ? (
                    <>
                      <Loader2 size={14} className="animate-spin" />
                      <span>Inference...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={14} />
                      <span>Prediksi Kelas</span>
                    </>
                  )}
                </button>
              </div>

              {predictionResult && (
                <div className="p-4 bg-zinc-50 border border-zinc-200 rounded-xl space-y-3">
                  <div className="text-xs font-bold text-zinc-800">Hasil Prediksi Multi-Model:</div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {Object.entries(predictionResult.predictions || {}).map(([model, info]: any) => (
                      <div key={model} className="p-3 bg-white border border-zinc-200 rounded-lg shadow-xs">
                        <div className="text-[10px] font-mono font-bold uppercase text-zinc-400">
                          {model}
                        </div>
                        <div className="text-sm font-bold text-emerald-700 mt-0.5 capitalize">
                          {info.top_class}
                        </div>
                        <div className="text-[11px] text-zinc-500 font-mono mt-0.5">
                          Conf: {(info.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
