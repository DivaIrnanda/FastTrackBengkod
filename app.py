import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main {
    background-color: #0f0f14;
}

[data-testid="stAppViewContainer"] {
    background: #0f0f14;
}

[data-testid="stSidebar"] {
    display: none !important;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e8e8f0;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #6b6b88;
    margin-bottom: 2rem;
}

.accent { color: #7c6aff; }

.card {
    background: #15151e;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.result-positive {
    background: linear-gradient(135deg, #1a0a2e, #2d1060);
    border: 1px solid #7c6aff;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-negative {
    background: linear-gradient(135deg, #0a1a0a, #0d3322);
    border: 1px solid #2ecc71;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.result-desc {
    font-size: 0.95rem;
    color: #9999bb;
}

.feature-tag {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #3a3a5a;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.82rem;
    color: #9999cc;
    margin: 3px;
    font-family: 'Space Mono', monospace;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #5555aa;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.feature-card {
    background: #15151e;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}

.feature-card:hover {
    border-color: #7c6aff;
}

.feature-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: #e8e8f0;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.feature-desc {
    font-size: 0.88rem;
    color: #9999bb;
    line-height: 1.6;
    margin-bottom: 0.5rem;
}

.feature-meta {
    font-size: 0.79rem;
    color: #6b6b88;
    line-height: 1.5;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-right: 6px;
}

.badge-red    { background: #2d0a1a; color: #ff6b9d; border: 1px solid #6b1c3b; }
.badge-orange { background: #2d1a0a; color: #ffaa6b; border: 1px solid #6b3a1c; }
.badge-green  { background: #0a2d10; color: #6bffaa; border: 1px solid #1c6b30; }
.badge-blue   { background: #0a1a2d; color: #6baeff; border: 1px solid #1c3b6b; }
.badge-purple { background: #1a0a2d; color: #c084fc; border: 1px solid #4a1c6b; }

.stButton > button {
    background: linear-gradient(135deg, #7c6aff, #5b50cc) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    width: 100%;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124, 106, 255, 0.35) !important;
}

div[data-testid="metric-container"] {
    background: #15151e;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 1rem;
}

hr {
    border-color: #2a2a3a !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path("best_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None

model = load_model()

# ─────────────────────────────────────────────
# FEATURE METADATA (LENGKAP)
# ─────────────────────────────────────────────
FEATURE_INFO = {
    'Have you ever had suicidal thoughts ?': {
        'type': 'categorical',
        'options': ['No', 'Yes'],
        'desc': 'Apakah pernah memiliki pikiran untuk bunuh diri',
        'icon': '💭',
        'importance': 'critical',
        'badge': 'badge-red',
        'badge_label': '🔴 Pengaruh Tertinggi',
        'detail': """
Pikiran untuk mengakhiri hidup (suicidal ideation) merupakan **salah satu gejala paling kuat** 
dalam mendeteksi depresi klinis.

**Mengapa penting?**  
Menurut penelitian, individu yang pernah mengalami pikiran bunuh diri memiliki risiko depresi 
**3–5× lebih tinggi** dibandingkan yang tidak. Ini bukan pertanyaan yang menghakimi, melainkan 
indikator kesehatan mental yang sangat penting untuk dipantau.

**Cara pengisian:**  
- **No** → Tidak pernah memiliki pikiran bunuh diri  
- **Yes** → Pernah memiliki pikiran tersebut (kapanpun dalam hidup)
""",
    },
    'Academic Pressure': {
        'type': 'slider',
        'min': 0, 'max': 5, 'default': 3,
        'desc': 'Tingkat tekanan akademik yang dirasakan (0 = tidak ada, 5 = sangat tinggi)',
        'icon': '📚',
        'importance': 'high',
        'badge': 'badge-orange',
        'badge_label': '🟠 Korelasi Tinggi',
        'detail': """
Tekanan akademik mengukur **seberapa berat beban studi** yang dirasakan mahasiswa, seperti 
tuntutan nilai, deadline tugas, ekspektasi dosen, atau kompetisi antar mahasiswa.

**Skala:**  
- **0** → Tidak ada tekanan sama sekali  
- **1–2** → Tekanan ringan, masih terasa nyaman  
- **3** → Tekanan moderat (rata-rata mahasiswa)  
- **4** → Tekanan berat, sering merasa kewalahan  
- **5** → Tekanan ekstrem, hampir tidak tertahankan  

**Dampak terhadap depresi:**  
Setiap peningkatan 1 poin tekanan akademik berhubungan dengan peningkatan risiko depresi 
sekitar **12–18%**. Mahasiswa dengan tekanan akademik ≥ 4 jauh lebih rentan.
""",
    },
    'Financial Stress': {
        'type': 'slider',
        'min': 1, 'max': 5, 'default': 3,
        'desc': 'Tingkat stres finansial/keuangan (1 = rendah, 5 = sangat tinggi)',
        'icon': '💰',
        'importance': 'high',
        'badge': 'badge-orange',
        'badge_label': '🟠 Korelasi Tinggi',
        'detail': """
Stres finansial mencerminkan **tekanan ekonomi** yang dialami mahasiswa, meliputi kekhawatiran 
biaya kuliah, kebutuhan hidup sehari-hari, utang, atau ketidakstabilan keuangan keluarga.

**Skala:**  
- **1** → Tidak ada kekhawatiran finansial  
- **2** → Sedikit khawatir tapi masih terkendali  
- **3** → Cukup khawatir, kadang mengganggu konsentrasi  
- **4** → Sangat khawatir, sering menjadi pikiran  
- **5** → Stres ekstrem, sangat mengganggu kehidupan  

**Konteks:**  
Stres finansial adalah salah satu pemicu depresi yang paling umum pada mahasiswa. 
Kekhawatiran tentang uang dapat mengganggu tidur, konsentrasi, dan motivasi belajar.
""",
    },
    'Age': {
        'type': 'number',
        'min': 18, 'max': 40, 'default': 22,
        'desc': 'Usia mahasiswa dalam tahun',
        'icon': '🎂',
        'importance': 'moderate',
        'badge': 'badge-blue',
        'badge_label': '🔵 Pengaruh Moderat',
        'detail': """
Usia mahasiswa (dalam tahun) berperan dalam menentukan pola kerentanan terhadap depresi. 
Mahasiswa berusia **18–22 tahun** sering berada dalam fase transisi kehidupan yang rentan 
(masuk dunia perkuliahan, berpisah dari keluarga, mandiri pertama kali).

**Rentang yang valid:** 18–40 tahun (untuk populasi mahasiswa)

**Pola yang ditemukan:**  
- Mahasiswa **tahun pertama** (usia 18–19) cenderung lebih rentan stres adaptasi  
- Mahasiswa **tingkat akhir** (usia 21–23) rentan karena tekanan skripsi & karier  
- Mahasiswa **lebih tua** (24+) bisa memiliki tantangan berbeda (kerja sambil kuliah, keluarga)
""",
    },
    'Work/Study Hours': {
        'type': 'slider',
        'min': 0, 'max': 24, 'default': 8,
        'desc': 'Jumlah jam kerja atau belajar per hari',
        'icon': '⏱️',
        'importance': 'moderate',
        'badge': 'badge-blue',
        'badge_label': '🔵 Pengaruh Moderat',
        'detail': """
Mengukur **total jam per hari** yang dihabiskan untuk kegiatan belajar dan/atau bekerja. 
Ini mencerminkan keseimbangan antara produktivitas dan waktu istirahat.

**Skala:** 0–24 jam per hari

**Rekomendasi WHO:**  
- Mahasiswa idealnya belajar **6–9 jam/hari** (termasuk kelas)  
- Di atas **12 jam/hari** mulai berisiko kelelahan (burnout)  
- Di atas **16 jam/hari** sangat berisiko terhadap kesehatan mental  

**Catatan:** Jam yang sangat rendah (0–2 jam) juga bisa menjadi tanda kehilangan motivasi, 
yang merupakan gejala depresi.
""",
    },
    'Dietary Habits': {
        'type': 'categorical',
        'options': ['Healthy', 'Moderate', 'Unhealthy'],
        'desc': 'Kebiasaan pola makan sehari-hari',
        'icon': '🥗',
        'importance': 'lifestyle',
        'badge': 'badge-green',
        'badge_label': '🟢 Faktor Gaya Hidup',
        'detail': """
Pola makan mencerminkan **kualitas nutrisi** sehari-hari mahasiswa, yang secara langsung 
memengaruhi fungsi otak dan suasana hati (mood).

**Kategori:**  
- **Healthy** → Makan teratur, banyak sayur/buah, protein cukup, minim junk food  
- **Moderate** → Kadang sehat kadang tidak, masih ada beberapa makanan bergizi  
- **Unhealthy** → Sering skip makan, dominan junk food/instan, minim nutrisi  

**Hubungan dengan mental health:**  
Kekurangan zat gizi seperti **omega-3, vitamin B12, zat besi, dan magnesium** terbukti 
berhubungan dengan peningkatan risiko depresi. Pola makan buruk sering kali menjadi 
lingkaran setan dengan depresi: depresi → malas makan → nutrisi buruk → memperparah depresi.
""",
    },
    'Study Satisfaction': {
        'type': 'slider',
        'min': 0, 'max': 5, 'default': 3,
        'desc': 'Tingkat kepuasan terhadap proses pembelajaran (0 = tidak puas, 5 = sangat puas)',
        'icon': '😊',
        'importance': 'moderate',
        'badge': 'badge-blue',
        'badge_label': '🔵 Pengaruh Moderat',
        'detail': """
Kepuasan belajar mengukur **seberapa bermakna dan memuaskan** proses perkuliahan yang dijalani 
mahasiswa, termasuk kepuasan terhadap materi, metode pengajaran, dan pencapaian akademik.

**Skala:**  
- **0** → Sangat tidak puas, merasa tidak ada gunanya kuliah  
- **1–2** → Kurang puas, banyak hal yang tidak sesuai harapan  
- **3** → Cukup puas, netral  
- **4** → Puas, menikmati proses belajar  
- **5** → Sangat puas, kuliah terasa bermakna dan menyenangkan  

**Hubungan dengan depresi:**  
Mahasiswa dengan kepuasan belajar rendah (0–2) berisiko lebih tinggi mengalami perasaan 
tidak berdaya dan kehilangan tujuan, yang merupakan gejala utama depresi.
""",
    },
    'Family History of Mental Illness': {
        'type': 'categorical',
        'options': ['No', 'Yes'],
        'desc': 'Riwayat penyakit mental dalam keluarga',
        'icon': '👨‍👩‍👧',
        'importance': 'moderate',
        'badge': 'badge-blue',
        'badge_label': '🔵 Faktor Risiko Genetik',
        'detail': """
Riwayat penyakit mental dalam keluarga mencerminkan **faktor genetik dan lingkungan keluarga** 
yang dapat meningkatkan kerentanan seseorang terhadap gangguan mental.

**Yang dimaksud riwayat mental:**  
Meliputi anggota keluarga inti (orang tua, saudara kandung) atau keluarga besar yang pernah 
didiagnosis kondisi seperti depresi, kecemasan, gangguan bipolar, skizofrenia, dll.

**Fakta ilmiah:**  
- Depresi memiliki **heritabilitas sekitar 37–50%** (berdasarkan studi kembar)  
- Anak dari orang tua dengan depresi memiliki risiko **2–3× lebih tinggi**  
- Faktor genetik berinteraksi dengan lingkungan (stress-diathesis model)

**Cara pengisian:**  
- **No** → Tidak ada riwayat penyakit mental di keluarga  
- **Yes** → Ada anggota keluarga dengan riwayat gangguan mental
""",
    },
    'CGPA': {
        'type': 'number_float',
        'min': 0.0, 'max': 10.0, 'default': 7.5,
        'desc': 'Indeks prestasi kumulatif (skala 0–10)',
        'icon': '🎓',
        'importance': 'moderate',
        'badge': 'badge-purple',
        'badge_label': '🟣 Indikator Akademik',
        'detail': """
CGPA (Cumulative Grade Point Average) adalah **rata-rata nilai akademik kumulatif** mahasiswa 
selama masa studi, pada skala 0–10.

**Interpretasi nilai:**  
- **8.5–10.0** → Sangat baik / cum laude  
- **7.0–8.4** → Baik  
- **5.5–6.9** → Cukup  
- **< 5.5** → Kurang / mengalami kesulitan akademik  

**Hubungan dengan depresi:**  
CGPA yang sangat rendah bisa menjadi konsekuensi **sekaligus penyebab** depresi:  
- Depresi menyebabkan kesulitan konsentrasi → nilai turun  
- Nilai rendah meningkatkan kecemasan dan stres → memperparah depresi  

Namun CGPA yang **sangat tinggi** juga bisa mengindikasikan perfeksionisme berlebihan 
yang berisiko terhadap kecemasan dan burnout.
""",
    },
    'Sleep Duration': {
        'type': 'categorical',
        'options': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
        'desc': 'Durasi tidur per malam',
        'icon': '😴',
        'importance': 'lifestyle',
        'badge': 'badge-green',
        'badge_label': '🟢 Faktor Gaya Hidup',
        'detail': """
Durasi tidur mengukur **berapa jam rata-rata tidur per malam**, yang sangat erat kaitannya 
dengan kesehatan mental dan fungsi kognitif.

**Kategori:**  
- **Less than 5 hours** → Kurang tidur berat; sangat berisiko  
- **5–6 hours** → Kurang tidur sedang; di bawah rekomendasi  
- **7–8 hours** → Ideal; sesuai rekomendasi WHO untuk dewasa muda  
- **More than 8 hours** → Tidur berlebihan; bisa menjadi gejala depresi  

**Fakta ilmiah:**  
- Kurang tidur kronis meningkatkan kadar kortisol (hormon stres) dan mengganggu regulasi emosi  
- Insomnia adalah **gejala sekaligus penyebab** depresi  
- Tidur berlebihan (hypersomnia) juga merupakan gejala depresi yang sering diabaikan  
- **7–8 jam** per malam adalah target optimal untuk kesehatan mental mahasiswa
""",
    },
    'Gender': {
        'type': 'categorical',
        'options': ['Male', 'Female'],
        'desc': 'Jenis kelamin',
        'icon': '👤',
        'importance': 'demographic',
        'badge': 'badge-purple',
        'badge_label': '🟣 Faktor Demografis',
        'detail': """
Jenis kelamin berperan dalam pola ekspresi dan prevalensi depresi, berkaitan dengan 
perbedaan **biologis, hormonal, dan sosial-budaya**.

**Perbedaan yang ditemukan di literatur:**  
- **Perempuan** secara statistik memiliki prevalensi depresi yang lebih tinggi (~2× dari pria)  
- **Laki-laki** cenderung lebih jarang mencari bantuan (stigma sosial) dan lebih sulit terdeteksi  
- Perbedaan hormonal (estrogen, progesteron) memengaruhi mood dan kerentanan depresi  

**Catatan model:**  
Gender memiliki pengaruh moderat (importance: 0.038). Artinya model tidak mendiskriminasi 
secara berlebihan berdasarkan gender, melainkan menggunakannya sebagai salah satu dari banyak faktor.
""",
    },
    'City': {
        'type': 'categorical',
        'options': [
            'Ahmedabad', 'Agra', 'Bangalore', 'Bhopal', 'Chennai',
            'Delhi', 'Faridabad', 'Ghaziabad', 'Hyderabad', 'Indore',
            'Jaipur', 'Kalyan', 'Kanpur', 'Kolkata', 'Lucknow',
            'Ludhiana', 'Meerut', 'Mumbai', 'Nagpur', 'Nashik',
            'Patna', 'Pune', 'Rajkot', 'Srinagar', 'Surat',
            'Thane', 'Vadodara', 'Varanasi', 'Vasai-Virar', 'Visakhapatnam'
        ],
        'desc': 'Kota tempat tinggal mahasiswa',
        'icon': '🏙️',
        'importance': 'low',
        'badge': 'badge-blue',
        'badge_label': '🔵 Faktor Lingkungan',
        'detail': """
Kota tempat tinggal mencerminkan **konteks lingkungan** mahasiswa, termasuk akses layanan 
kesehatan mental, tingkat kemacetan, polusi, biaya hidup, dan tekanan sosial kota besar vs kecil.

**Dataset mencakup 30 kota besar di India:**  
Mumbai, Delhi, Bangalore, Kolkata, Chennai, Hyderabad, Pune, dan kota-kota besar lainnya.

**Pengaruh kota terhadap kesehatan mental:**  
- Kota **metropolitan besar** (Mumbai, Delhi) → biaya hidup tinggi, kompetisi ketat, stres lebih tinggi  
- Namun akses layanan kesehatan mental juga lebih baik  
- **Tingkat kebisingan dan polusi** di kota besar berhubungan dengan peningkatan kecemasan  

**Importance score:** 0.021 (terendah dari 12 fitur)  
Kota memiliki pengaruh paling kecil dalam model, menunjukkan bahwa faktor internal (pikiran, 
tekanan akademik, finansial) jauh lebih dominan daripada lokasi geografis.
""",
    },
}

ORDINAL_ENCODE = {
    'Have you ever had suicidal thoughts ?': {'No': 0, 'Yes': 1},
    'Dietary Habits': {'Healthy': 0, 'Moderate': 1, 'Unhealthy': 2},
    'Family History of Mental Illness': {'No': 0, 'Yes': 1},
    'Sleep Duration': {'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3},
    'Gender': {'Male': 0, 'Female': 1},
    'City': {c: i for i, c in enumerate(sorted([
        'Ahmedabad', 'Agra', 'Bangalore', 'Bhopal', 'Chennai',
        'Delhi', 'Faridabad', 'Ghaziabad', 'Hyderabad', 'Indore',
        'Jaipur', 'Kalyan', 'Kanpur', 'Kolkata', 'Lucknow',
        'Ludhiana', 'Meerut', 'Mumbai', 'Nagpur', 'Nashik',
        'Patna', 'Pune', 'Rajkot', 'Srinagar', 'Surat',
        'Thane', 'Vadodara', 'Varanasi', 'Vasai-Virar', 'Visakhapatnam'
    ]))}
}

SELECTED_FEATURES = [
    'Have you ever had suicidal thoughts ?',
    'Academic Pressure',
    'Financial Stress',
    'Age',
    'Work/Study Hours',
    'Dietary Habits',
    'Study Satisfaction',
    'Family History of Mental Illness',
    'CGPA',
    'Sleep Duration',
    'Gender',
    'City',
]

# ─────────────────────────────────────────────
# NAVIGATION (TAB-BASED, NO SIDEBAR)
# ─────────────────────────────────────────────
st.markdown('<p class="hero-title">🧠 MindCheck <span class="accent">Student</span></p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Sistem Prediksi Depresi Mahasiswa berbasis Machine Learning</p>', unsafe_allow_html=True)

tab_prediksi, tab_fitur = st.tabs(["🏠 Prediksi", "ℹ️ Penjelasan Fitur"])

# ─────────────────────────────────────────────
# TAB: PREDIKSI
# ─────────────────────────────────────────────
with tab_prediksi:
    st.markdown("---")

    if model is None:
        st.warning("⚠️ File model `best_model.pkl` tidak ditemukan. Pastikan file model tersedia di direktori yang sama.")
        st.info("💡 Jalankan notebook terlebih dahulu dan simpan model dengan: `joblib.dump(best_model, 'best_model.pkl')`")

    col_form, col_result = st.columns([1.2, 1], gap="large")

    with col_form:
        st.markdown('<p class="section-label">📝 Input Data Mahasiswa</p>', unsafe_allow_html=True)

        input_values = {}

        # Row 1
        c1, c2 = st.columns(2)
        with c1:
            input_values['Gender'] = st.selectbox("👤 Gender", FEATURE_INFO['Gender']['options'])
            input_values['Age'] = st.number_input("🎂 Usia", min_value=18, max_value=40, value=22)
        with c2:
            input_values['City'] = st.selectbox("🏙️ Kota", FEATURE_INFO['City']['options'])
            input_values['CGPA'] = st.number_input("🎓 CGPA (0–10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)

        st.markdown("")

        # Row 2
        c3, c4 = st.columns(2)
        with c3:
            input_values['Sleep Duration'] = st.selectbox("😴 Durasi Tidur", FEATURE_INFO['Sleep Duration']['options'])
            input_values['Dietary Habits'] = st.selectbox("🥗 Pola Makan", FEATURE_INFO['Dietary Habits']['options'])
        with c4:
            input_values['Have you ever had suicidal thoughts ?'] = st.selectbox(
                "💭 Pikiran Bunuh Diri?", FEATURE_INFO['Have you ever had suicidal thoughts ?']['options'])
            input_values['Family History of Mental Illness'] = st.selectbox(
                "👨‍👩‍👧 Riwayat Mental Keluarga", FEATURE_INFO['Family History of Mental Illness']['options'])

        st.markdown("")

        # Sliders
        st.markdown('<p class="section-label">Skala Tekanan & Kepuasan</p>', unsafe_allow_html=True)
        input_values['Academic Pressure'] = st.slider("📚 Tekanan Akademik", 0, 5, 3)
        input_values['Financial Stress'] = st.slider("💰 Stres Finansial", 1, 5, 3)
        input_values['Study Satisfaction'] = st.slider("😊 Kepuasan Belajar", 0, 5, 3)
        input_values['Work/Study Hours'] = st.slider("⏱️ Jam Belajar", 0, 24, 8)

        st.markdown("")
        predict_btn = st.button("🔍 Prediksi Sekarang", use_container_width=True)

    with col_result:
        st.markdown('<p class="section-label">📈 Hasil Prediksi</p>', unsafe_allow_html=True)

        if predict_btn:
            if model is None:
                st.error("Model belum tersedia. Silakan tambahkan file `best_model.pkl`.")
            else:
                # Encode input
                encoded = {}
                for feat in SELECTED_FEATURES:
                    val = input_values[feat]
                    if feat in ORDINAL_ENCODE:
                        encoded[feat] = ORDINAL_ENCODE[feat][val]
                    else:
                        encoded[feat] = val

                input_df = pd.DataFrame([encoded])[SELECTED_FEATURES]

                prediction = model.predict(input_df)[0]
                prob = None
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(input_df)[0]
                elif hasattr(model, 'decision_function'):
                    df_score = model.decision_function(input_df)[0]
                    prob_pos = 1 / (1 + np.exp(-df_score))
                    prob = [1 - prob_pos, prob_pos]

                if prediction == 1:
                    st.markdown("""
                    <div class="result-positive">
                        <div class="result-label" style="color:#c084fc;">⚠️ Terdeteksi Depresi</div>
                        <div class="result-desc">Berdasarkan data yang dimasukkan, mahasiswa ini <b>berpotensi mengalami depresi</b>. 
                        Disarankan untuk berkonsultasi dengan konselor atau tenaga profesional kesehatan mental.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-negative">
                        <div class="result-label" style="color:#2ecc71;">✅ Tidak Terdeteksi Depresi</div>
                        <div class="result-desc">Berdasarkan data yang dimasukkan, mahasiswa ini <b>tidak menunjukkan tanda-tanda depresi</b>. 
                        Tetap jaga kesehatan mental dan gaya hidup seimbang.</div>
                    </div>
                    """, unsafe_allow_html=True)

                if prob is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="section-label">Probabilitas Prediksi</p>', unsafe_allow_html=True)

                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.metric("Tidak Depresi", f"{prob[0]*100:.1f}%")
                    with col_p2:
                        st.metric("Depresi", f"{prob[1]*100:.1f}%")

                    # Bar chart probabilitas
                    fig, ax = plt.subplots(figsize=(5, 1.8))
                    fig.patch.set_facecolor('#15151e')
                    ax.set_facecolor('#15151e')
                    ax.barh(['Tidak Depresi', 'Depresi'],
                            [prob[0]*100, prob[1]*100],
                            color=['#2ecc71', '#c084fc'], height=0.5, edgecolor='none')
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('%', color='#6b6b88', fontsize=8)
                    ax.tick_params(colors='#9999bb', labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                # Input summary
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="section-label">Ringkasan Input</p>', unsafe_allow_html=True)
                display_values = {
                    k: (f"{v:.2f}" if k == 'CGPA' else str(v))
                    for k, v in input_values.items()
                }
                summary_data = {k: [v] for k, v in display_values.items()}
                df_summary = pd.DataFrame(summary_data).T.rename(columns={0: 'Nilai'})
                st.dataframe(df_summary, use_container_width=True, height=280)

        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding: 3rem 1.5rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
                <div style="color: #5555aa; font-family: 'Space Mono', monospace; font-size: 0.85rem;">
                    Isi form di kiri<br>lalu klik <b style="color:#7c6aff">Prediksi Sekarang</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#4a4a6a; text-align:center;">
    ⚠️ Aplikasi ini bersifat edukatif dan tidak menggantikan diagnosis profesional kesehatan mental.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB: PENJELASAN FITUR (DETAIL)
# ─────────────────────────────────────────────
with tab_fitur:
    st.markdown("---")
    st.markdown('<p class="hero-title" style="font-size:1.8rem;">ℹ️ Penjelasan <span class="accent">Fitur</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Deskripsi lengkap, konteks ilmiah, dan panduan pengisian setiap fitur yang digunakan model</p>', unsafe_allow_html=True)


    st.markdown('<p class="section-label">📋 Detail Setiap Fitur (klik untuk memperluas)</p>', unsafe_allow_html=True)

    for feat in SELECTED_FEATURES:
        info = FEATURE_INFO[feat]

        with st.expander(f"{info['icon']}  **{feat}**"):
            # Badge
            st.markdown(
                f'<span class="badge {info["badge"]}">{info["badge_label"]}</span>',
                unsafe_allow_html=True
            )
            st.markdown("")

            # Description
            st.markdown(f"**📌 Deskripsi Singkat:**  \n{info['desc']}")
            st.markdown("")

            # Type info
            if info['type'] == 'categorical':
                st.markdown("**🏷️ Tipe Data:** Kategorikal")
                opts_html = " ".join([f'<span class="feature-tag">{o}</span>' for o in info['options']])
                st.markdown(f"**📂 Pilihan yang tersedia:**<br>{opts_html}", unsafe_allow_html=True)
            elif info['type'] == 'slider':
                st.markdown(f"**🏷️ Tipe Data:** Numerik Diskrit (Skala {info['min']}–{info['max']})")
            elif info['type'] in ['number', 'number_float']:
                st.markdown(f"**🏷️ Tipe Data:** Numerik (Rentang {info['min']}–{info['max']})")

            st.markdown("")
            # Detailed explanation
            st.markdown("**📖 Penjelasan Lengkap:**")
            st.markdown(info['detail'])

    st.markdown("---")
    st.markdown("""
    <div style="background:#15151e; border:1px solid #2a2a3a; border-radius:12px; padding:1.4rem 1.6rem; font-size:0.88rem; color:#9999bb; line-height:2.0;">

    <div style="font-size:1rem; font-family:'Space Mono',monospace; color:#e8e8f0; margin-bottom:1rem;">🤖 Bagaimana Cara Kerja Aplikasi Ini?</div>

    Aplikasi ini menggunakan <b style="color:#c084fc;">kecerdasan buatan (AI)</b> untuk membantu
    mengenali kemungkinan depresi pada mahasiswa. Sederhananya, AI ini sudah
    "belajar" dari data ribuan mahasiswa, lalu bisa mengenali pola yang biasanya
    muncul pada orang yang mengalami depresi —
    mirip seperti dokter yang sudah menangani banyak pasien dan mulai hafal gejalanya.<br><br>

    <div style="font-size:0.88rem; font-family:'Space Mono',monospace; color:#7c6aff; margin-bottom:0.4rem;">📊 Data yang Digunakan</div>
    AI ini dilatih dari data lebih dari <b style="color:#e8e8f0;">27.000 mahasiswa</b> di berbagai kota di India.
    Setiap mahasiswa menjawab pertanyaan seputar kehidupan sehari-hari, tekanan studi, pola tidur,
    dan kondisi mentalnya. Semua jawaban itu dipakai untuk "mengajari" AI cara mengenali tanda-tanda depresi.<br><br>

    <div style="font-size:0.88rem; font-family:'Space Mono',monospace; color:#7c6aff; margin-bottom:0.4rem;">✅ Seberapa Akurat?</div>
    Kami mencoba 5 metode AI yang berbeda, lalu memilih yang paling akurat.
    AI terbaik berhasil menebak dengan benar sekitar <b style="color:#e8e8f0;">85 dari 100 kasus</b>.
    Artinya cukup bisa diandalkan sebagai alat bantu awal — tapi tetap bukan pengganti dokter atau psikolog.<br><br>

    <div style="font-size:0.88rem; font-family:'Space Mono',monospace; color:#7c6aff; margin-bottom:0.4rem;">📝 Apa yang Dianalisis?</div>
    AI ini memperhatikan <b style="color:#e8e8f0;">12 hal</b> tentang kamu: mulai dari seberapa tertekan di kampus,
    pola tidur, pola makan, nilai akademik (CGPA), hingga riwayat kesehatan mental keluarga.
    Semua informasi itu digabungkan untuk menghasilkan perkiraan.<br><br>

    <div style="background:#1a1a2e; border-left:3px solid #7c6aff; padding:0.8rem 1rem; border-radius:0 8px 8px 0; font-size:0.83rem; color:#8888aa;">
    ⚠️ <b style="color:#9999bb;">Catatan Penting:</b> Hasil dari aplikasi ini hanyalah
    <b style="color:#c084fc;">perkiraan awal</b>, bukan diagnosis medis.
    Jika kamu atau orang di sekitarmu merasa butuh bantuan,
    jangan ragu untuk menghubungi konselor kampus atau tenaga kesehatan mental yang terpercaya.
    </div>

    </div>
    """, unsafe_allow_html=True)
