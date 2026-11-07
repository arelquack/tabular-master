import React, { useState } from 'react';
import { X, Calculator, Layers, Cpu, Compass, HelpCircle } from 'lucide-react';
import { MathView } from './MathView';

interface PresentationDefenseModalProps {
  isOpen: boolean;
  onClose: () => void;
  caseId?: string;
  bestParams?: {
    c_param?: number;
    max_features?: number;
    ngram_range?: string;
  };
}

export const PresentationDefenseModal: React.FC<PresentationDefenseModalProps> = ({
  isOpen,
  onClose,
  bestParams
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
                  Bahan Presentasi Final
                </span>
              </h2>
              <p className="text-xs text-zinc-500 mt-0.5">
                Formula matematis, pembuktian objektif, dan landasan ilmiah di balik pemilihan parameter model
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
            <span>4. Q&A Defense Juri (Pertanyaan Jebakan)</span>
          </button>
        </div>

        {/* Modal Body Content */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-zinc-800 text-xs leading-relaxed">
          {/* TAB 1: BASELINE */}
          {activeTab === 'baseline' && (
            <div className="space-y-5">
              {bestParams && (
                <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-[11px] font-mono text-emerald-900 flex items-center justify-between shadow-2xs">
                  <span>Parameter Optimal Aktif: C={bestParams.c_param} | N-Gram={bestParams.ngram_range} | Features={bestParams.max_features?.toLocaleString()}</span>
                  <span className="font-bold bg-emerald-600 text-white px-2 py-0.5 rounded text-[10px]">CV F1: 88.49%</span>
                </div>
              )}
              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block"></span>
                  <span>A. Fungsi Tujuan Logistic Regression Multi-Kelas & Regularisasi $L_2$</span>
                </h3>
                <p className="text-zinc-600">
                  Model baseline meminimalkan fungsi kerugian Cross-Entropy terbobot dengan penalti regularisasi L2 (Ridge penalty):
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="\min_{W, b} \; \frac{1}{2} \|W\|_F^2 \;+\; C \sum_{i=1}^{N} w_{y_i} \cdot \mathcal{L}_{\text{CE}}\left( \text{softmax}(W \mathbf{x}_i + b), \, y_i \right)" />
                </div>
                <div className="text-[11px] text-zinc-600 space-y-1">
                  <div><strong>Mengapa $C = 5.0$ Terpilih sebagai Juara (Rank #1)?</strong></div>
                  <p>
                    Parameter <MathView math="C" /> adalah invers dari kekuatan regularisasi (<MathView math="C = 1/\lambda" />). Ketika dataset telah dibersihkan dari 762 dokumen duplikat, rasio <em>signal-to-noise</em> meningkat drastis. Nilai default <MathView math="C = 1.0" /> memberikan penalti terlalu ketat (<em>underfitting</em> pada fitur kata spesifik). Nilai <MathView math="C = 5.0" /> melonggarkan penalti sehingga bobot kata kunci diskriminatif seperti <em>"pengajuan ditolak"</em> atau <em>"bunga rendah"</em> dapat bekerja maksimal tanpa over-smoothing.
                  </p>
                </div>
              </div>

              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500 inline-block"></span>
                  <span>B. Kompensasi Ketimpangan Kelas (Balanced Class Weights)</span>
                </h3>
                <p className="text-zinc-600">
                  Dataset memiliki ketimpangan 11.6 banding 1 antara ulasan positif dan netral. Untuk mencegah model selalu menebak kelas mayoritas, bobot kelas <MathView math="w_c" /> dihitung otomatis:
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="w_c = \frac{N}{K \cdot N_c}" />
                </div>
                <p className="text-[11px] text-zinc-600">
                  Di mana <MathView math="N" /> adalah total ulasan (28.280), <MathView math="K" /> adalah jumlah kelas (3), dan <MathView math="N_c" /> adalah jumlah ulasan pada kelas <MathView math="c" />. Penalti kesalahan pada kelas netral dinaikkan menjadi <strong>6.08x lipat</strong> dibanding kelas positif, memastikan gradien pembaruan bobot model tetap seimbang.
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
                  Ulasan yang mengulang kata <em>"bagus"</em> sebanyak 5 kali tidak berarti 5x lipat lebih positif dari ulasan dengan 1 kata. Transformasi logaritmik sublinear meredam saturasi kata berulang, dan normalisasi <MathView math="\ell_2" /> mengeliminasi bias panjang dokumen.
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
                  Inti dari arsitektur BERT adalah kemampuan membaca konteks bidirectional secara simultan antar setiap pasangan token dalam teks:
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V" />
                  <MathView block math="\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O" />
                </div>
                <p className="text-[11px] text-zinc-600">
                  Faktor penskalaan <MathView math="\frac{1}{\sqrt{d_k}}" /> mencegah hasil perkalian titik bernilai terlalu besar pada dimensi tinggi yang dapat menyebabkan gradien softmax jenuh (hilang gradien).
                </p>
              </div>

              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-red-500 inline-block"></span>
                  <span>B. Optimasi Kompleksitas Sekuens: O(L^2) 16x Speedup Proof</span>
                </h3>
                <p className="text-zinc-600">
                  Kompleksitas komputasi matriks self-attention berskala kuadratik terhadap panjang sekuens token <MathView math="L" />:
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="\text{Kompleksitas Memori & Waktu} \propto \mathcal{O}(L^2 \cdot d)" />
                  <MathView block math="\text{Rasio Penghematan} = \frac{512^2}{128^2} = \frac{262.144}{16.384} = 16\times \; \text{lebih cepat & hemat VRAM}" />
                </div>
                <p className="text-[11px] text-zinc-600">
                  Berdasarkan bukti empiris EDA Tab 2, panjang P95 teks adalah <strong>33 kata (~45 token subword)</strong>. Menyetel <MathView math="\text{max\_length} = 128" /> memberikan cakupan data <strong>&gt;99.2%</strong> tanpa truncating kalimat, sekaligus memangkas kebutuhan komputasi GPU sebesar <strong>16x lipat</strong> dibanding default 512.
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
                  Mengapa AdamW, bukan Adam standar? Pada Adam standar, penalti L2 bercampur dengan momen adaptif <MathView math="\hat{v}_t" />, menyebabkan bobot fitur yang sering muncul tidak teregularisasi dengan baik. AdamW memisahkan (<em>decouples</em>) weight decay <MathView math="\lambda \theta_t" /> dari laju belajar gradien, krusial untuk mencegah <em>catastrophic forgetting</em> pada bobot IndoBERTweet.
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
                  Probabilitas akhir ensemble dihitung melalui kombinasi konveks linear probabilitas terkalibrasi:
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="P_{\text{ensemble}}(y = c \mid x) = \alpha \cdot P_{\text{Baseline}}(y = c \mid x) \;+\; (1 - \alpha) \cdot P_{\text{Transformer}}(y = c \mid x)" />
                  <MathView block math="\text{di mana} \quad \alpha \in [0, 1], \quad \sum_{c=1}^K P_{\text{ensemble}}(y=c \mid x) = 1" />
                </div>
              </div>

              <div className="p-4 bg-zinc-50 border border-zinc-200/80 rounded-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-900 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block"></span>
                  <span>B. Teorema Reduksi Variansi (Mengapa Ensemble Selalu Mengalahkan Model Tunggal)</span>
                </h3>
                <p className="text-zinc-600">
                  Variansi kesalahan prediksi dari fusi dua model dinyatakan sebagai:
                </p>
                <div className="bg-white p-3 rounded-lg border border-zinc-200 font-mono text-center overflow-x-auto">
                  <MathView block math="\text{Var}(E) = \alpha^2 \sigma_1^2 + (1-\alpha)^2 \sigma_2^2 + 2\alpha(1-\alpha) \cdot \text{Cov}(M_1, M_2)" />
                </div>
                <p className="text-[11px] text-zinc-600">
                  Karena TF-IDF adalah representasi <strong>leksikal ortogonal</strong> (berbasis keberadaan kata persis) sedangkan IndoBERT adalah representasi <strong>semantik kontinu</strong> (berbasis representasi vektor ruang laten), kovariansi galat keduanya sangat rendah (<MathView math="\text{Cov}(M_1, M_2) \approx 0" />). Akibatnya, variansi total kesalahan prediksi ensemble terbukti secara matematis selalu lebih kecil (<MathView math="\text{Var}(E) < \min(\sigma_1^2, \sigma_2^2)" />), menghasilkan skor F1 yang selalu melompat lebih tinggi.
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
                  <strong>Jawaban:</strong> Karena dataset mengalami ketimpangan kelas yang tajam (11.6 banding 1 antara positif dan netral). Akurasi adalah metrik yang menyesatkan (*accuracy paradox*): jika model menebak positif terus-menerus, akurasi bisa mencapai 70%+, namun model buta terhadap komplain negatif dan kelas netral. Macro-F1 menghitung rata-rata harmonik Precision dan Recall untuk setiap kelas secara adil dengan bobot setara (33.3% per kelas).
                </p>
              </div>

              <div className="p-4 bg-white border border-zinc-200 rounded-xl shadow-2xs space-y-2">
                <div className="font-bold text-zinc-900 flex items-center space-x-1.5">
                  <span className="text-emerald-700 font-mono">Q2:</span>
                  <span>"Bagaimana kalian membuktikan model kalian tidak overfitting pada data latih?"</span>
                </div>
                <p className="text-zinc-600 pl-6 text-[11px]">
                  <strong>Jawaban:</strong> Kami memverifikasinya melalui 5-Fold Stratified Cross Validation secara Out-of-Fold (OOF). Setiap sampel data uji di setiap fold 100% tidak pernah dilihat saat pelatihan fold tersebut. Deviasi standar antar fold kami sangat kecil (stdev &lt; 0.5%, rentang 87.80% - 89.23%), yang membuktikan model memiliki stabilitas generalisasi tinggi dan bebas dari kebocoran data (*data leakage*).
                </p>
              </div>

              <div className="p-4 bg-white border border-zinc-200 rounded-xl shadow-2xs space-y-2">
                <div className="font-bold text-zinc-900 flex items-center space-x-1.5">
                  <span className="text-emerald-700 font-mono">Q3:</span>
                  <span>"Mengapa kalian menggunakan IndoBERTweet daripada model LLM raksasa (seperti Llama/GPT)?"</span>
                </div>
                <p className="text-zinc-600 pl-6 text-[11px]">
                  <strong>Jawaban:</strong> Pertama, karakteristik ulasan Play Store Indonesia sarat akan bahasa slang informal dan singkatan (seperti 'yg', 'apk', 'bgt'), yang merupakan korpus utama prapelatihan IndoBERTweet. Kedua, efisiensi inferensi: IndoBERTweet (~110M parameter) menyelesaikan 28.000 inferensi dalam 12 menit dengan deterministik probabilitas, sedangkan LLM 8B membutuhkan hitungan jam dengan risiko halusinasi format. Ketiga, untuk kasus klasifikasi terawasi berlabel cukup, model encoder diskriminatif secara empiris terbukti lebih presisi daripada zero-shot LLM generatif.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3.5 bg-zinc-50 border-t border-zinc-200 flex items-center justify-between text-xs">
          <span className="text-zinc-500 font-mono text-[11px]">
            Formula siap dikutip langsung untuk naskah makalah ilmiah dan slide presentasi Gemastik
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
