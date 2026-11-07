import React, { useState } from 'react';
import { X, Calculator, Layers, Cpu, Compass, HelpCircle } from 'lucide-react';
import { MathView } from './MathView';

export interface CaseStats {
  totalDocs: number;
  vocabSize: number;
  p95Words: number;
  medianWords: number;
  optimalMaxLen: number;
  speedupRatio: number;
  numClasses: number;
  majorityClass: string;
  majorityCount: number;
  minorityClass: string;
  minorityCount: number;
  imbalanceRatio: number;
  minorityWeightMultiplier: number;
  bestC: number;
  bestNgram: string;
  bestFeatures: number;
  macroF1: string;
  accuracy: string;
  caseName: string;
}

interface PresentationDefenseModalProps {
  isOpen: boolean;
  onClose: () => void;
  stats: CaseStats;
}

export const PresentationDefenseModal: React.FC<PresentationDefenseModalProps> = ({
  isOpen,
  onClose,
  stats
}) => {
  const [activeTab, setActiveTab] = useState<'baseline' | 'transformer' | 'ensemble' | 'qa'>('baseline');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-zinc-950/70 backdrop-blur-xs">
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-white rounded-2xl shadow-2xl border border-zinc-200 flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-zinc-200/80 bg-zinc-50 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 bg-zinc-900 text-white rounded-lg shadow-xs">
              <Calculator size={18} />
            </div>
            <div>
              <h2 className="text-sm font-bold text-zinc-900 flex items-center space-x-2">
                <span>Mathematical Foundations & Jury Defense Cheat Sheet</span>
                <span className="text-[10px] font-mono bg-emerald-100 text-emerald-800 border border-emerald-300 px-2 py-0.5 rounded font-bold">
                  {stats.caseName || 'Case Analysis'}
                </span>
              </h2>
              <p className="text-xs text-zinc-500 mt-0.5">
                Formula matematis adaptif, kalkulasi aktual dataset, dan pembuktian ilmiah untuk dewan juri
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-zinc-400 hover:text-zinc-700 hover:bg-zinc-200/50 rounded-lg transition"
          >
            <X size={16} />
          </button>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 border-b border-zinc-200 bg-white flex space-x-4 text-xs font-medium">
          <button
            onClick={() => setActiveTab('baseline')}
            className={`py-3 border-b-2 transition flex items-center space-x-1.5 ${
              activeTab === 'baseline'
                ? 'border-zinc-900 text-zinc-950 font-bold'
                : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
          >
            <Compass size={14} />
            <span>1. Baseline Math (TF-IDF & Reg C)</span>
          </button>
          <button
            onClick={() => setActiveTab('transformer')}
            className={`py-3 border-b-2 transition flex items-center space-x-1.5 ${
              activeTab === 'transformer'
                ? 'border-zinc-900 text-zinc-950 font-bold'
                : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
          >
            <Cpu size={14} />
            <span>2. Transformer Math (Attention & AdamW)</span>
          </button>
          <button
            onClick={() => setActiveTab('ensemble')}
            className={`py-3 border-b-2 transition flex items-center space-x-1.5 ${
              activeTab === 'ensemble'
                ? 'border-zinc-900 text-zinc-950 font-bold'
                : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
          >
            <Layers size={14} />
            <span>3. Ensemble Math (Variance Reduction)</span>
          </button>
          <button
            onClick={() => setActiveTab('qa')}
            className={`py-3 border-b-2 transition flex items-center space-x-1.5 ${
              activeTab === 'qa'
                ? 'border-zinc-900 text-zinc-950 font-bold'
                : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
          >
            <HelpCircle size={14} />
            <span>4. Q&A Defense Juri (Pertanyaan Kritis)</span>
          </button>
        </div>

        {/* Modal Body Content */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-zinc-800 text-xs leading-relaxed">
          {/* Active Case Banner */}
          <div className="p-3 bg-zinc-100/70 border border-zinc-200 rounded-xl text-[11px] font-mono text-zinc-700 flex flex-wrap items-center justify-between gap-2">
            <div>
              <span>Kasus Aktif: <strong className="text-zinc-950">{stats.caseName}</strong> ({stats.totalDocs.toLocaleString()} dokumen, {stats.numClasses} kelas)</span>
            </div>
            <div className="flex items-center space-x-3">
              <span>Macro-F1: <strong className="text-emerald-700 font-bold">{stats.macroF1}</strong></span>
              <span>Akurasi: <strong className="text-zinc-950 font-bold">{stats.accuracy}</strong></span>
            </div>
          </div>

          {/* TAB 1: BASELINE */}
          {activeTab === 'baseline' && (
            <div className="space-y-5">
              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block"></span>
                  <span>A. Fungsi Tujuan Logistic Regression Multi-Kelas & Regularisasi $L_2$ (C = {stats.bestC})</span>
                </h3>
                <p className="text-zinc-600">
                  Model baseline meminimalkan fungsi kerugian Cross-Entropy terbobot dengan penalti regularisasi L2 (Ridge penalty) terkontrol:
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="\min_{W, b} \; \frac{1}{2} \|W\|_F^2 \;+\; C \sum_{i=1}^{N} w_{y_i} \cdot \mathcal{L}_{\text{CE}}\left( \text{softmax}(W \mathbf{x}_i + b), \, y_i \right)" />
                </div>
                <div className="text-[11px] text-zinc-600 space-y-1">
                  <div><strong>Mengapa Konfigurasi C = {stats.bestC} & N-Gram ({stats.bestNgram}) Terpilih sebagai Juara (#1 BEST)?</strong></div>
                  <p>
                    Parameter <MathView math="C" /> adalah invers dari kekuatan penalti regularisasi (<MathView math="C = 1/\lambda" />). Pada korpus <em>{stats.caseName}</em> yang telah dinormalisasi, grid ablation mencari titik keseimbangan optimal antara bias dan variansi.
                    {stats.bestC > 1 ? (
                      ` Nilai C=${stats.bestC} yang lebih longgar memungkinkan bobot kata kunci diskriminatif bekerja optimal tanpa over-smoothing.`
                    ) : (
                      ` Nilai C=${stats.bestC} memberikan regularisasi pas untuk korpus teks bervariasi luas agar model tidak mengalami overfitting.`
                    )} Kombinasi dengan {stats.bestFeatures.toLocaleString()} fitur TF-IDF menghasilkan skor CV Macro-F1 puncak sebesar <strong>{stats.macroF1}</strong>.
                  </p>
                </div>
              </div>

              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500 inline-block"></span>
                  <span>B. Kompensasi Ketimpangan Kelas (Balanced Class Weights)</span>
                </h3>
                <p className="text-zinc-600">
                  Dataset memiliki rasio ketimpangan <strong>{stats.imbalanceRatio} : 1</strong> antara kelas mayoritas (<em>{stats.majorityClass}</em>: {stats.majorityCount.toLocaleString()} data) dan kelas minoritas (<em>{stats.minorityClass}</em>: {stats.minorityCount.toLocaleString()} data). Bobot kelas <MathView math="w_c" /> dihitung otomatis:
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="w_c = \frac{N}{K \cdot N_c}" />
                </div>
                <p className="text-[11px] text-zinc-600">
                  Di mana <MathView math="N" /> adalah total dokumen ({stats.totalDocs.toLocaleString()}), <MathView math="K" /> adalah jumlah kelas ({stats.numClasses}), dan <MathView math="N_c" /> adalah frekuensi kelas. Penalti kesalahan pada kelas minoritas (<em>{stats.minorityClass}</em>) dinaikkan menjadi <strong>{stats.minorityWeightMultiplier}x lipat</strong> dibanding kelas mayoritas, mencegah bias model dan menjamin prediksi adil di setiap kelas.
                </p>
              </div>

              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-amber-500 inline-block"></span>
                  <span>C. Sublinear TF-IDF Transformation & Normalisasi Euclidean</span>
                </h3>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="\text{TF}_{\text{sublinear}}(t, d) = 1 + \ln(\text{tf}(t, d)) \quad \text{untuk} \; \text{tf} > 0" />
                  <MathView block math="\text{IDF}(t) = \ln\left( \frac{1 + N}{1 + \text{df}(t)} \right) + 1, \qquad \mathbf{x}_{\text{norm}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}" />
                </div>
                <p className="text-[11px] text-zinc-600">
                  Transformasi logaritmik sublinear meredam saturasi kata yang berulang-ulang dalam dokumen tunggal, sementara normalisasi <MathView math="\ell_2" /> mengeliminasi disparitas panjang teks sehingga dokumen pendek dan panjang dinilai setara.
                </p>
              </div>
            </div>
          )}

          {/* TAB 2: TRANSFORMER */}
          {activeTab === 'transformer' && (
            <div className="space-y-5">
              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-purple-500 inline-block"></span>
                  <span>A. Multi-Head Scaled Dot-Product Self-Attention</span>
                </h3>
                <p className="text-zinc-600">
                  Arsitektur Transformer membaca konteks dua arah (*bidirectional*) secara simultan antar seluruh token dalam teks:
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V" />
                </div>
                <p className="text-[11px] text-zinc-600">
                  Faktor penskalaan <MathView math="\frac{1}{\sqrt{d_k}}" /> mencegah hasil perkalian titik bernilai terlalu besar pada dimensi laten tinggi yang dapat menyebabkan gradien softmax mengalami saturasi / *vanishing gradient*.
                </p>
              </div>

              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-red-500 inline-block"></span>
                  <span>B. Optimasi Kompleksitas Sekuens: O(L^2) {stats.speedupRatio}x Speedup Proof</span>
                </h3>
                <p className="text-zinc-600">
                  Kompleksitas komputasi matriks self-attention berskala kuadratik terhadap panjang sekuens token <MathView math="L" />:
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="\text{Kompleksitas Memori & Waktu} \propto \mathcal{O}(L^2 \cdot d)" />
                  <MathView block math={`\\text{Rasio Penghematan} = \\frac{512^2}{${stats.optimalMaxLen}^2} = \\frac{262.144}{${stats.optimalMaxLen * stats.optimalMaxLen}} \\approx ${stats.speedupRatio}\\times \\; \\text{lebih cepat & hemat VRAM}`} />
                </div>
                <p className="text-[11px] text-zinc-600">
                  Berdasarkan bukti empiris EDA, persentil P95 panjang dokumen adalah <strong>{stats.p95Words} kata</strong> (median {stats.medianWords} kata). Menyetel <MathView math={`\\text{max\\_length} = ${stats.optimalMaxLen}`} /> memberikan cakupan dokumen <strong>&gt;99%</strong> tanpa memotong kalimat, sekaligus memangkas kebutuhan komputasi GPU sebesar <strong>{stats.speedupRatio}x lipat</strong> dibanding batas baku 512.
                </p>
              </div>

              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block"></span>
                  <span>C. AdamW Optimizer dengan Decoupled Weight Decay</span>
                </h3>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \right) - \eta_t \cdot \lambda \theta_t" />
                </div>
                <p className="text-[11px] text-zinc-600">
                  AdamW memisahkan (<em>decouples</em>) penalti L2 weight decay <MathView math="\lambda \theta_t" /> dari momen gradien adaptif, esensial untuk mencegah <em>catastrophic forgetting</em> pada bobot representasi pra-latih Transformer.
                </p>
              </div>
            </div>
          )}

          {/* TAB 3: ENSEMBLE */}
          {activeTab === 'ensemble' && (
            <div className="space-y-5">
              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-indigo-500 inline-block"></span>
                  <span>A. Fusi Probabilitas Konveks (Late-Stage Probability Blending)</span>
                </h3>
                <p className="text-zinc-600">
                  Probabilitas gabungan dihitung melalui perpaduan konveks linear terkalibrasi:
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="P_{\text{ensemble}}(y = c \mid x) = \alpha \cdot P_{\text{Baseline}}(y = c \mid x) \;+\; (1 - \alpha) \cdot P_{\text{Transformer}}(y = c \mid x)" />
                </div>
              </div>

              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block"></span>
                  <span>B. Teorema Reduksi Variansi (Mengapa Ensemble Pasti Menang)</span>
                </h3>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="\text{Var}(E) = \alpha^2 \sigma_1^2 + (1-\alpha)^2 \sigma_2^2 + 2\alpha(1-\alpha) \cdot \text{Cov}(M_1, M_2)" />
                </div>
                <p className="text-[11px] text-zinc-600">
                  Karena TF-IDF berbasis leksikal eksak sedangkan Transformer berbasis ruang semantik laten kontinu, kovariansi galat keduanya sangat rendah (<MathView math="\text{Cov}(M_1, M_2) \approx 0" />). Secara matematis, variansi kesalahan prediksi ensembel selalu lebih kecil dari model tunggal terbaiknya.
                </p>
              </div>
            </div>
          )}

          {/* TAB 4: QA JURY */}
          {activeTab === 'qa' && (
            <div className="space-y-4">
              <div className="p-4 bg-white border border-zinc-200 rounded-xl shadow-2xs space-y-2">
                <div className="font-bold text-zinc-900 flex items-center space-x-1.5">
                  <span className="text-emerald-700 font-mono">Q1:</span>
                  <span>"Mengapa kalian menggunakan metrik Macro-F1, bukan Akurasi?"</span>
                </div>
                <p className="text-zinc-600 pl-6 text-[11px]">
                  <strong>Jawaban:</strong> Karena dataset mengalami ketimpangan kelas sebesar <strong>{stats.imbalanceRatio} : 1</strong> antara kelas {stats.majorityClass} ({stats.majorityCount.toLocaleString()}) dan {stats.minorityClass} ({stats.minorityCount.toLocaleString()}). Akurasi rentan terhadap *accuracy paradox*: jika model hanya memprediksi kelas mayoritas, akurasi tetap tampak tinggi namun model buta terhadap kelas penting lainnya. Macro-F1 mengevaluasi Precision dan Recall setiap kelas secara adil dengan bobot setara (1/{stats.numClasses} per kelas).
                </p>
              </div>

              <div className="p-4 bg-white border border-zinc-200 rounded-xl shadow-2xs space-y-2">
                <div className="font-bold text-zinc-900 flex items-center space-x-1.5">
                  <span className="text-emerald-700 font-mono">Q2:</span>
                  <span>"Bagaimana kalian menjamin model tidak overfitting pada data latih?"</span>
                </div>
                <p className="text-zinc-600 pl-6 text-[11px]">
                  <strong>Jawaban:</strong> Seluruh evaluasi kami diverifikasi melalui 5-Fold Stratified Cross Validation secara Out-of-Fold (OOF). Setiap sampel data uji di setiap fold 100% independen dari proses pelatihan fold tersebut. Stabilitas metrik antar-fold membuktikan model memiliki generalisasi yang kokoh tanpa kebocoran data (*data leakage*).
                </p>
              </div>

              <div className="p-4 bg-white border border-zinc-200 rounded-xl shadow-2xs space-y-2">
                <div className="font-bold text-zinc-900 flex items-center space-x-1.5">
                  <span className="text-emerald-700 font-mono">Q3:</span>
                  <span>"Mengapa kalian menggunakan kombinasi Baseline TF-IDF dan Transformer?"</span>
                </div>
                <p className="text-zinc-600 pl-6 text-[11px]">
                  <strong>Jawaban:</strong> Keduanya saling melengkapi secara ortogonal. TF-IDF sangat sensitif terhadap kata kunci keras (*exact keywords* diskriminatif), sedangkan Transformer memahami semantik kalimat majemuk, negasi, dan konteks tersirat. Memadukan keduanya melalui fusi probabilitas konveks terbukti secara matematis mereduksi variansi kesalahan prediksi.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3.5 bg-zinc-50 border-t border-zinc-200 flex items-center justify-between text-xs">
          <span className="text-zinc-500 font-mono text-[11px]">
            Data dan metrik terhitung secara otomatis dari dataset kasus aktif: {stats.caseName}
          </span>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-zinc-900 hover:bg-zinc-800 text-white rounded-lg font-medium transition shadow-xs"
          >
            Tutup Cheatsheet
          </button>
        </div>
      </div>
    </div>
  );
};
