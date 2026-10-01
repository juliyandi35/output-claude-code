import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

# =============================================================================
md(r"""
# 🔥 Hands-on: Model Hurdle Spasio-Temporal untuk Hotspot Kebakaran Kalimantan
### Belajar metode tesis langkah demi langkah — tanpa perlu latar belakang statistik

> **Sumber:** tesis *"Model Hurdle Spasio-Temporal Berbasis INLA untuk Hotspot Kebakaran Hutan dan Lahan di Kalimantan dengan Efek Lag Iklim"*.
> **Isi notebook ini:** semua tahap metode tesis dijalankan ulang pada **data simulasi** (jadi kebenarannya kita ketahui dan bisa dicek), disertai bagan alur dan puluhan visualisasi.

---
### ⚠️ Baca ini dulu (kejujuran metode)
| Hal | Penjelasan |
|---|---|
| **Data** | Data di sini **simulasi** mirip Kalimantan, *bukan* data FIRMS/NASA POWER asli. Alasannya: dengan data simulasi kita tahu "jawaban benar"-nya, sehingga bisa melihat apakah model berhasil menemukannya. |
| **INLA** | Tesis memakai paket R-**INLA**. INLA tidak ada di Python, jadi di sini kita memakai **inti yang sama** (model Gauss laten + aproksimasi Laplace + hiperparameter ditetapkan) yang ditulis sendiri di `hurdle_lab.py`. Kode R INLA yang sebenarnya ada di bagian akhir (tidak dijalankan). |
| **Angka** | Angka hasil di sini **tidak akan sama** dengan tesis. Yang kita tiru adalah **logika dan urutan metodenya**. Angka asli tesis selalu ditandai *"(tesis)"*. |

### 🗺️ Peta perjalanan
| Bagian | Pertanyaan yang dijawab |
|---|---|
| 0 | Bagan alur: data masuk lewat mana, keluar jadi apa? |
| 1 | Bagaimana titik-titik api satelit menjadi tabel *sel × bulan*? |
| 2 | Kenapa datanya "aneh": banyak nol **dan** ekor sangat panjang? |
| 3 | Kenapa Poisson gagal, binomial negatif (NB) lebih baik, dan hurdle lebih baik lagi? |
| 4 | Komponen "laten": ruang (BYM2), waktu (AR(1)), musim (RW1 siklik) |
| 5 | Efek iklim berlag (hujan 0–3 bulan lalu) |
| 6 | Komponen bersama (*copy*) antara bagian kejadian & jumlah |
| 7 | Inti INLA: aproksimasi Laplace, dijelaskan dengan contoh mini |
| 8 | Membagi data latih/uji **menurut waktu** & melatih 6 model |
| 9 | Menilai prediksi: CRPS, PIT, kalibrasi, nol, total bulanan, bulan ekstrem |
| 10 | Apakah selisih antar model nyata? (selisih berpasangan + bootstrap blok) |
| 11 | Membaca hasil: efek iklim, peta ruang, musim, waktu |
| 12 | Rangkuman, kesalahan umum, latihan mandiri, kode R-INLA asli |

> 💡 **Cara belajar:** jalankan sel dari atas ke bawah. Di banyak tempat ada kotak **🧪 Coba sendiri** — ubah angkanya dan jalankan ulang untuk merasakan efeknya.
""")

code(r"""
import sys, warnings, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.special import expit, gammaln
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
from hurdle_lab import *
from flow import draw_flow
rng = np.random.default_rng(2026)
print("Siap. Semua pustaka termuat.")
""")

# =============================================================================
md(r"""
---
# BAGIAN 0 — Bagan alur data lengkap
Sebelum masuk detail, lihat dulu **peta besarnya**: dari titik api yang direkam satelit sampai kesimpulan akhir.
Setiap kotak punya nomor; nomor yang sama dipakai di bagian-bagian notebook di bawah.

**Cara membaca:** biru = data mentah → kuning = pengolahan data → hijau = pemodelan → merah muda = evaluasi → ungu = hasil.
""")
code(r"""
fig = draw_flow("alur_data_lengkap.png")
plt.show()
""")
md(r"""
### Ringkasan alur dalam bahasa sehari-hari
1. **Kumpulkan** catatan titik api dari satelit (FIRMS) dan catatan cuaca harian (NASA POWER).
2. **Bersihkan**: ambil hanya deteksi yang meyakinkan, hanya di vegetasi Kalimantan; ubah cuaca harian jadi bulanan; buat "keterlambatan" (hujan 1, 2, 3 bulan lalu).
3. **Kotak-kotakkan** pulau jadi petak 60 × 60 km; **hitung** berapa titik api di setiap petak setiap bulan → *tabel panel* (23.232 baris).
4. **Pisahkan** data lama (untuk belajar) dan data baru (untuk ujian) — **menurut waktu**, bukan acak.
5. **Pecah** hitungan jadi dua pertanyaan: *"ada api atau tidak?"* dan *"kalau ada, berapa banyak?"* (inilah **hurdle**).
6. **Latih** model yang memahami musim, tren waktu, tetangga, dan cuaca (dengan INLA).
7. **Ujikan** pada data baru memakai skor probabilistik, lalu **bandingkan** model secara adil.
8. **Baca** hasilnya: model mana lebih baik, dan *di mana* dan *kapan* ia lebih baik; pola ruang, musim, dan waktu apa yang ditemukan.
""")

# =============================================================================
md(r"""
---
# BAGIAN 1 — Dari titik api satelit ke tabel *sel × bulan* (kotak 1–3)

### Konsep
Satelit mencatat **titik** api (koordinat tepat). Tetapi tesis tidak memodelkan titik; ia memodelkan **jumlah titik per petak per bulan**.
Analogi: bukan mencatat posisi setiap hujan jatuh, melainkan **berapa mm hujan per kecamatan per bulan**.

Kita buat dunia simulasi: pulau kecil berbentuk Kalimantan, **55 petak**, **108 bulan**.
""")
code(r"""
d = simulate(seed=2026, T=108)
xy, W, n, T = d["xy"], d["W"], d["n"], d["T"]
Y = d["Y"]            # matriks T x n : hitungan hotspot
print(f"Jumlah petak (sel): {n}   Jumlah bulan: {T}   Baris panel: {n*T}")
print(f"Total hotspot: {Y.sum():,}")
print(f"Tetangga queen: rerata {W.sum(1).mean():.2f} per sel  (tesis: 6,95)")
""")
md(r"""
### 1.1 Dari titik ke petak
Ambil satu bulan puncak. Kita **sebar titik-titik api** secara acak di dalam setiap petak sesuai hitungannya (agar terlihat bagaimana titik berubah jadi angka per petak).
""")
code(r"""
t_peak = int(np.argmax(Y.sum(1)))
rng2 = np.random.default_rng(1)
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
# (a) titik mentah
for i in range(n):
    k = min(int(Y[t_peak, i]), 150)   # batasi agar gambar tidak penuh
    px = xy[i, 0] + rng2.uniform(-.5, .5, k); py = xy[i, 1] + rng2.uniform(-.5, .5, k)
    ax[0].scatter(px, py, s=3, c="#d62728", alpha=.5)
ax[0].set_title("① Titik hotspot mentah\n(1 titik = 1 deteksi satelit)")
# (b) grid + batas
for i in range(n):
    ax[1].add_patch(plt.Rectangle((xy[i,0]-.5, xy[i,1]-.5), 1, 1, fc="#f2f2f2", ec="k", lw=.7))
    pts = xy[i] + rng2.uniform(-.4,.4,(min(int(Y[t_peak,i]),60),2)); ax[1].scatter(pts[:,0], pts[:,1], s=2, c="#d62728", alpha=.5)
ax[1].set_title("② Tumpuk kotak 60 km (grid)\n(petak dengan daratan <10% dibuang)")
# (c) hitungan
sc = ax[2].scatter(xy[:,0], xy[:,1], c=Y[t_peak]+1, s=900, marker="s", cmap="YlOrRd", norm=LogNorm())
for i in range(n): ax[2].text(xy[i,0], xy[i,1], int(Y[t_peak,i]), ha="center", va="center", fontsize=7)
ax[2].set_title("③ Hitung titik per petak\n= satu bulan dari tabel panel")
for a in ax: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
plt.colorbar(sc, ax=ax[2], label="hitungan + 1 (skala log)", shrink=.7)
plt.tight_layout(); plt.show()
""")
md(r"""
### 1.2 Tabel panel
Diulang untuk setiap bulan, hasilnya adalah tabel panjang. **Satu baris = satu petak di satu bulan.** Inilah data yang dimodelkan.
""")
code(r"""
panel = pd.DataFrame({
    "sel": np.tile(np.arange(n), T), "bulan_ke": np.repeat(np.arange(1, T+1), n),
    "bulan_kalender": np.repeat(d["month"] + 1, n),
    "hotspot": Y.reshape(-1),
    "hujan_lag0": d["X_raw"]["hujan_lag0"].reshape(-1), "hujan_lag1": d["X_raw"]["hujan_lag1"].reshape(-1),
    "hujan_lag2": d["X_raw"]["hujan_lag2"].reshape(-1), "hujan_lag3": d["X_raw"]["hujan_lag3"].reshape(-1),
    "suhu": d["X_raw"]["suhu"].reshape(-1), "RH": d["X_raw"]["kelembapan"].reshape(-1)}).round(1)
print(f"Panel: {panel.shape[0]:,} baris (tesis: 23.232 = 176 sel × 132 bulan)")
panel.sample(8, random_state=3)
""")

# =============================================================================
md(r"""
---
# BAGIAN 2 — Mengenal data: kenapa "aneh"? (kotak 3 → analisis deskriptif)

Tesis menemukan **tiga sifat** yang sulit ditangani satu distribusi sekaligus:
1. **Banyak nol** (tesis: 39,94% sel-bulan nol)
2. **Nolnya bukan acak**, melainkan musiman dan berbeda antarwilayah
3. **Ekor sangat berat**: median hanya 5, tapi maksimum 11.334

Kita cek ketiganya di data simulasi.
""")
code(r"""
y = Y.reshape(-1)
pos = y[y > 0]
print(f"Proporsi nol            : {np.mean(y==0)*100:5.1f}%   (tesis 39,94%)")
print(f"Rerata hitungan         : {y.mean():6.2f}   (tesis 35,59)")
print(f"Rasio ragam / rerata    : {y.var()/y.mean():6.1f}   (tesis 1.766,6; Poisson menuntut = 1)")
print(f"Median hitungan positif : {np.median(pos):6.1f}   (tesis 5)")
print(f"Kuantil 99% (positif)   : {np.percentile(pos,99):6.0f}   (tesis ≈965)")
print(f"Maksimum                : {y.max():6d}   (tesis 11.334)")
tot_m = Y.sum(1); top6 = np.sort(tot_m)[::-1][:6].sum()/tot_m.sum()
print(f"6 bulan teratas = {top6*100:.1f}% dari semua hotspot  (tesis 58,2%)")
""")
code(r"""
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
# (a) histogram log
ax[0,0].hist(np.log1p(y), bins=60, color="#4c78a8")
ax[0,0].set_title("(a) Sebaran log(1+hitungan)\nmassa tinggi di nol + ekor panjang ke kanan")
ax[0,0].set_xlabel("log(1 + hitungan)"); ax[0,0].set_ylabel("jumlah sel-bulan")
# (b) proporsi nol per bulan + total
zero_m = (Y == 0).mean(1)
ax[0,1].bar(range(T), tot_m, color="#bbbbbb", label="total hotspot")
ax[0,1].set_yscale("log"); ax[0,1].set_ylabel("total hotspot (log)")
a2 = ax[0,1].twinx(); a2.plot(range(T), zero_m, "r-", label="proporsi nol"); a2.set_ylim(0,1); a2.grid(False)
a2.set_ylabel("proporsi nol", color="r")
ax[0,1].set_title("(b) Nol turun saat api memuncak:\nnol itu MUSIMAN, bukan acak"); ax[0,1].set_xlabel("bulan ke-")
# (c) musiman kalender
mm = pd.DataFrame({"bln": np.repeat(d["month"]+1, n), "y": y})
box = [mm.y[mm.bln==b].values for b in range(1,13)]
ax[1,0].boxplot([np.log1p(b) for b in box], labels=range(1,13), showfliers=False)
ax[1,0].set_title("(c) Pola musiman: puncak sekitar bulan 8–9 (Agu–Sep)")
ax[1,0].set_xlabel("bulan kalender"); ax[1,0].set_ylabel("log(1+hitungan)")
# (d) Lorenz: konsentrasi
srt = np.sort(y)[::-1]; cum = np.cumsum(srt)/srt.sum()
ax[1,1].plot(np.arange(1, len(y)+1)/len(y)*100, cum*100, lw=2)
ax[1,1].plot([0,100],[0,100],"k--",lw=.8)
ax[1,1].axvline(10, color="r", ls=":"); ax[1,1].text(11, 30, f"10% sel-bulan teratas\n= {cum[int(.1*len(y))]*100:.0f}% hotspot", color="r")
ax[1,1].set_title("(d) Konsentrasi: segelintir sel-bulan menyumbang mayoritas"); ax[1,1].set_xlabel("% sel-bulan (urut dari terbesar)"); ax[1,1].set_ylabel("% kumulatif hotspot")
plt.tight_layout(); plt.show()
""")
md(r"""
### 💡 Poin penting: ini **bukan** "pencilan"
Banyak orang menyebut ini *"banyak nol dan banyak pencilan"*. Tesis **mengoreksi** pembacaan itu:
- Nilai besar di bulan kering **bukan galat** — itu bagian sah dari data dan justru yang **terpenting** untuk pemantauan.
- Menghapus atau "meredamnya" akan membuang informasi yang paling dicari.
- Masalah sebenarnya adalah **dua hal berbeda yang tercampur**: *apakah ada deteksi?* dan *berapa banyak jika ada?*

### 2.2 Peta: nol juga bergantung lokasi
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
z_cell = (Y == 0).mean(0)
s1 = ax[0].scatter(xy[:,0], xy[:,1], c=z_cell, s=800, marker="s", cmap="Blues", vmin=0, vmax=1)
ax[0].set_title("Proporsi bulan TANPA hotspot per sel\n(mirip Gambar 4.4 tesis)"); plt.colorbar(s1, ax=ax[0], shrink=.8)
s2 = ax[1].scatter(xy[:,0], xy[:,1], c=np.log10(Y.sum(0)+1), s=800, marker="s", cmap="YlOrRd")
ax[1].set_title("Total hotspot per sel (log10)"); plt.colorbar(s2, ax=ax[1], shrink=.8)
for a in ax: a.set_aspect("equal"); a.grid(False); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
""")
md(r"""
### 2.3 Apakah sel bertetangga mirip? — Indeks Moran
Indeks Moran mengukur: *"sel yang bersebelahan, apakah nilainya mirip?"* Nilai > 0 = mirip (menggerombol). Tesis: selalu positif (0,105–0,861) → graf ketetanggaan masuk akal dipakai.
""")
code(r"""
def moran(x, W):
    z = x - x.mean()
    return len(x) / W.sum() * (z @ W @ z) / (z @ z)
mor = [moran(np.log1p(Y[t]), W) for t in range(T)]
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(mor, "o-", ms=3); ax.axhline(0, color="k", lw=.8)
ax.set_title(f"Indeks Moran bulanan: rentang {min(mor):.2f} – {max(mor):.2f} (tesis 0,105–0,861)"); ax.set_xlabel("bulan ke-"); ax.set_ylabel("Moran I")
plt.show()
""")

# =============================================================================
md(r"""
---
# BAGIAN 3 — Dari Poisson ke binomial negatif ke hurdle (kotak 5)

## 3.1 Poisson: terlalu "kaku"
**Poisson** hanya punya satu pengatur (rata-rata μ) dan memaksa **ragam = rata-rata**. Data kita punya ragam ≈ 360× rata-rata (tesis: 1.767×). Poisson pasti salah.

## 3.2 Binomial negatif (NB): lebih lentur
NB menambah satu pengatur, **ukuran k** (*size*). Makin kecil k, makin "melebar" (overdispersi): ragam = μ + μ²/k.
Peluang nol pada NB: **P(Y=0) = (k/(k+μ))^k**.

Geser parameternya dan lihat bentuknya:
""")
code(r"""
from scipy.stats import poisson, nbinom
def nb_pmf(y, mu, k):  # parameter: rerata mu dan ukuran k
    p = k / (k + mu); return nbinom.pmf(y, k, p)
ys = np.arange(0, 60)
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
mu = 10
ax[0].bar(ys-.2, poisson.pmf(ys, mu), .4, label="Poisson(10)", color="#4c78a8")
ax[0].bar(ys+.2, nb_pmf(ys, mu, 1.0), .4, label="NB(μ=10, k=1)", color="#e45756")
ax[0].set_title("Rerata sama (10), bentuk beda:\nNB punya ekor panjang"); ax[0].legend()
for k in [0.2, 0.5, 1, 5, 50]:
    ax[1].plot(ys, nb_pmf(ys, mu, k), label=f"k={k}")
ax[1].set_title("Makin kecil k → makin melebar\n(k→∞ menjadi Poisson)"); ax[1].legend(); ax[1].set_ylim(0, .25)
mus = np.linspace(.1, 40, 200)
for k in [0.2, 0.5, 1, 5]:
    ax[2].plot(mus, (k/(k+mus))**k, label=f"k={k}")
ax[2].set_title("P(Y=0) pada NB: turun bila μ naik\n— TETAPI terkunci ke μ dan k yang sama"); ax[2].set_xlabel("μ"); ax[2].legend()
plt.tight_layout(); plt.show()
""")
md(r"""
> **🧪 Coba sendiri:** ubah `mu = 10` jadi `2` atau `40`. Perhatikan bahwa pada μ kecil NB sudah menghasilkan banyak nol *sendirian* — itulah mengapa tesis menekankan bahwa lawan yang adil bagi hurdle adalah **NB berstruktur laten sama (ZN)**, bukan sekadar Poisson.

### Masalah NB tunggal
Satu NB **mengikat** peluang nol dan besarnya hitungan ke parameter yang sama. Padahal bisa jadi:
- *apakah ada deteksi* dikendalikan oleh satu hal (misal hujan bulan ini), dan
- *berapa banyak deteksi* dikendalikan hal lain (misal hujan 1–2 bulan lalu, kelembapan).

## 3.3 Zero-inflated vs hurdle
| | Zero-inflated (Lambert, 1992) | **Hurdle** (Mullahy, 1986) — dipakai tesis |
|---|---|---|
| Sumber nol | **Dua**: nol "struktural" + nol dari distribusi hitungan | **Satu**: semua nol dari "tidak melewati rintangan" |
| Hitungan positif | distribusi hitungan biasa | distribusi **terpotong** (hanya ≥ 1) |
| Cocok bila | nol struktural bisa dibedakan | tidak bisa dibedakan |

Alasan tesis memilih hurdle: pada hotspot satelit, nol bisa berarti *tidak ada api* **atau** *api tak terdeteksi (awan/asap/waktu lintasan)* — keduanya **tak dapat dibedakan dari data**, jadi nol struktural tidak teridentifikasi.

**Hurdle = dua langkah:**
```
Langkah 1 (kejadian):  Z = 1 jika ada deteksi (y>0), 0 jika tidak.   Z ~ Bernoulli(π)   → model LOGISTIK
Langkah 2 (jumlah)  :  jika Z=1, berapa banyak?  y−1 ~ Binomial Negatif(μ, k)           → model LOG
```
Di tesis, langkah 2 memakai **NB "geser"**: hitungan positif dikurangi 1 supaya bisa bernilai 0 (aproksimasi NB terpotong, lebih mudah diestimasi).
""")
code(r"""
# Visual: pohon keputusan hurdle + sebaran akhir
pi, mu_n, k_n = 0.6, 25, 0.8
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
ax[0].axis("off"); ax[0].set_xlim(0,10); ax[0].set_ylim(0,6)
def bx(x,y,t,c): ax[0].text(x,y,t,ha="center",va="center",fontsize=10,bbox=dict(boxstyle="round,pad=.4",fc=c,ec="k"))
bx(1.5,3,"Sel c, bulan t\n(satu baris panel)","#DCEBFA")
bx(5.2,4.7,f"Z = 0  → y = 0\npeluang 1−π = {1-pi:.2f}","#F9DDE3")
bx(5.2,1.3,f"Z = 1  (ada deteksi)\npeluang π = {pi:.2f}","#E3F4E1")
bx(8.6,1.3,f"y = 1 + NB(μ={mu_n}, k={k_n})\n(berapa banyak?)","#FFF1CC")
for (a,b,c,d_) in [(2.7,3.3,4.1,4.3),(2.7,2.7,4.1,1.7),(6.5,1.3,7.0,1.3)]:
    ax[0].annotate("",xy=(c,d_),xytext=(a,b),arrowprops=dict(arrowstyle="->"))
ax[0].set_title("Pohon keputusan hurdle (dua langkah)")
ys = np.arange(0, 120)
pmf = np.zeros(len(ys)); pmf[0] = 1 - pi
pmf[1:] = pi * nb_pmf(ys[1:] - 1, mu_n, k_n)
ax[1].bar(ys, pmf, color="#4c78a8"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-5, 1)
ax[1].set_title("Hasil: sebaran akhir = 'tonggak' besar di nol + ekor NB\n(sumbu y log)"); ax[1].set_xlabel("hitungan y"); ax[1].set_ylabel("peluang")
plt.tight_layout(); plt.show()
""")
md(r"""
### 3.4 NB "geser" vs NB terpotong eksak — seberapa beda?
Tesis mengakui ini aproksimasi. Mari lihat sendiri bedanya pada hitungan kecil.
""")
code(r"""
mu_n, k_n = 8, 1.0
ys = np.arange(1, 40)
shift = nb_pmf(ys - 1, mu_n, k_n)                         # NB(y-1)
trunc = nb_pmf(ys, mu_n, k_n) / (1 - nb_pmf(0, mu_n, k_n))  # NB terpotong di nol (eksak)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].bar(ys-.2, shift, .4, label="NB geser (y−1) — dipakai tesis"); ax[0].bar(ys+.2, trunc, .4, label="NB terpotong eksak")
ax[0].legend(); ax[0].set_title("Peluang tiap hitungan positif")
ax[1].plot(ys, shift - trunc, "o-"); ax[1].axhline(0, color="k", lw=.8)
ax[1].set_title("Selisih (geser − terpotong): paling besar di hitungan 1–3\n→ dampak pada hitungan kecil, diakui sbg keterbatasan"); ax[1].set_xlabel("y")
plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""
---
# BAGIAN 4 — Efek "laten": ruang, waktu, musim (kotak 6)

Hitungan sebuah sel bulan ini bukan hanya soal cuaca. Ada **kecenderungan tersembunyi (laten)** yang tak diukur langsung:
- **Ruang** — sel tertentu memang lebih rawan (gambut, aktivitas lahan), dan sel bertetangga cenderung mirip.
- **Waktu** — tahun kering (El Niño) menggeser semuanya, dan bulan ini mirip bulan lalu.
- **Musim** — Agustus–September selalu lebih panas/kering dari Februari.

Setiap komponen diberi *"sifat"* tertentu lewat struktur matematikanya. Kita lihat satu per satu.

## 4.1 Ruang: graf tetangga queen + BYM2
**Graf queen:** dua sel bertetangga bila bersentuhan di sisi **atau sudut** (maks. 8 tetangga).
""")
code(r"""
fig, ax = plt.subplots(figsize=(5.8, 5.6))
for i in range(n):
    for j in range(i+1, n):
        if W[i,j]: ax.plot(xy[[i,j],0], xy[[i,j],1], "-", c="#999", lw=.8, zorder=1)
ax.scatter(xy[:,0], xy[:,1], s=260, c="#4c78a8", zorder=2, edgecolor="k")
i0 = 20
nb_ = np.where(W[i0]>0)[0]
ax.scatter(xy[i0,0], xy[i0,1], s=420, c="#d62728", zorder=3, edgecolor="k")
ax.scatter(xy[nb_,0], xy[nb_,1], s=300, c="#f58518", zorder=3, edgecolor="k")
ax.set_title(f"Graf queen: merah = satu sel, oranye = {len(nb_)} tetangganya\n(rerata semua sel {W.sum(1).mean():.2f}; tesis 6,95)")
ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); plt.show()
""")
md(r"""
**BYM2** (Riebler dkk., 2016) memecah efek ruang menjadi dua bagian lalu mencampurnya dengan satu parameter **φ (0–1)**:

> efek ruang = σ · ( √φ · *[bagian terstruktur / mulus ala tetangga]* + √(1−φ) · *[bagian acak murni per sel]* )

- **φ → 1**: pola ruang **halus**, tetangga mirip. **φ → 0**: tiap sel berdiri sendiri.
- **σ** = seberapa besar amplitudo efek ruang.

Berikut contoh *sampel acak* dari BYM2 dengan φ berbeda (graf sama!):
""")
code(r"""
Qs = scale_icar(icar_Q(W)) + 1e-6*np.eye(n)
L = np.linalg.cholesky(np.linalg.inv(Qs))
rs = np.random.default_rng(5)
u_star = L @ rs.normal(size=n); u_star -= u_star.mean()
v = rs.normal(size=n)
fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
for a, phi in zip(ax, [0.0, 0.5, 0.9, 1.0]):
    eff = np.sqrt(phi)*u_star + np.sqrt(1-phi)*v
    sc = a.scatter(xy[:,0], xy[:,1], c=eff, s=520, marker="s", cmap="RdBu_r", vmin=-3, vmax=3)
    a.set_title(f"φ = {phi}"); a.set_aspect("equal"); a.grid(False); a.set_xticks([]); a.set_yticks([])
plt.colorbar(sc, ax=ax, shrink=.8, label="efek ruang"); plt.suptitle("BYM2: komponen acak yang sama, φ makin besar → pola makin mulus/bergerombol", y=1.02); plt.show()
""")
md(r"""
> **Di tesis** φ ≈ 0,93–0,95, artinya variasi ruang kebanyakan berasal dari komponen **terstruktur** (tetangga mirip). Tetapi selang kredibelnya sangat lebar (0,29–1,00) sehingga tesis memperingatkan: *jangan dibaca sebagai bukti kuat*.

## 4.2 Waktu: AR(1) — "bulan ini mirip bulan lalu"
Efek waktu: **u_t = ρ · u_{t−1} + kejutan acak**. Parameter **ρ** (0–1) = seberapa kuat ingatan bulan lalu.
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
r3 = np.random.default_rng(11); eps = r3.normal(size=108)
for rho in [0.0, 0.5, 0.8, 0.95]:
    u = np.zeros(108)
    for t in range(1,108): u[t] = rho*u[t-1] + np.sqrt(1-rho**2)*eps[t]
    ax[0].plot(u, label=f"ρ={rho}", lw=1.4)
ax[0].legend(ncol=4, fontsize=8); ax[0].set_title("Lintasan AR(1) dari 'kejutan' yang sama: ρ besar → lintasan mulus, ingatan panjang"); ax[0].set_xlabel("bulan")
lags = np.arange(0, 25)
for rho in [0.5, 0.8, 0.95]: ax[1].plot(lags, rho**lags, "o-", ms=3, label=f"ρ={rho}")
ax[1].set_title("Korelasi dengan bulan ke-h sebelumnya = ρ^h"); ax[1].set_xlabel("selisih bulan h"); ax[1].legend()
plt.tight_layout(); plt.show()
""")
md(r"""
> Di tesis ρ ≈ 0,55–0,78. Data agregat memang punya autokorelasi lag‑1 = 0,712.

## 4.3 Musim: random walk orde 1 **siklik** (12 bulan)
Setiap bulan kalender punya efek sendiri; bulan bersebelahan dipaksa mirip (selisihnya kecil), dan **Desember menyambung ke Januari** (siklik).
""")
code(r"""
tr = d["truth"]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
mn = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
ax[0].plot(mn, tr["season"], "o-", lw=2); ax[0].axhline(0, color="k", lw=.7)
ax[0].set_title("Profil musim (kebenaran simulasi): puncak Agu–Sep\n(tesis: +1,6/+1,7 pada Agu/Sep; terendah ≈ −0,85)")
# siklik
th = np.linspace(0, 2*np.pi, 13)
s13 = np.append(tr["season"], tr["season"][0])
ax[1].remove(); ax2 = fig.add_subplot(1,2,2, projection="polar")
ax2.plot(th, s13 - s13.min() + .3, "o-"); ax2.set_xticks(th[:-1]); ax2.set_xticklabels(mn); ax2.set_yticklabels([])
ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
ax2.set_title("Tampilan melingkar: Des tersambung ke Jan")
plt.tight_layout(); plt.show()
""")
md(r"""
## 4.4 Menyusun semuanya: dari efek ke peluang & jumlah
Prediktor linear (kotak 6) pada skala "bebas":
```
η_kejadian(c,t) = μ_z + [iklim] + a_c + u_t + s_m(t)          →  π = 1/(1+e^−η)      (logit → peluang)
η_jumlah (c,t)  = μ_n + [iklim] + b_a·a_c + b_u·u_t + b_s·s_m(t) + offset(log luas) → μ = e^η   (log → rerata)
```
Visualisasi **kontribusi tiap komponen** untuk satu sel terpilih di data simulasi:
""")
code(r"""
c0 = 12
fig, ax = plt.subplots(5, 1, figsize=(13, 10), sharex=True)
tt = np.arange(T)
ax[0].plot(tt, np.full(T, tr["a"][c0]), c="#4c78a8"); ax[0].set_ylabel("ruang a_c\n(konstan)")
ax[1].plot(tt, tr["u"], c="#e45756"); ax[1].set_ylabel("waktu u_t")
ax[2].plot(tt, tr["season"][d["month"]], c="#54a24b"); ax[2].set_ylabel("musim s_m(t)")
ax[3].plot(tt, d["eta_z"][:, c0], c="k"); ax[3].set_ylabel("η kejadian\n(total)")
ax[4].bar(tt, Y[:, c0], color="#bbb"); ax[4].set_yscale("symlog"); ax[4].set_ylabel("hitungan\nteramati")
ax[4].set_xlabel("bulan ke-"); ax[0].set_title(f"Satu sel (nomor {c0}): komponen laten menjumlah menjadi η, lalu menjadi data")
plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""
---
# BAGIAN 5 — Efek iklim berlag (kotak 2c & 6)

### Kenapa lag?
Gambut dan bahan bakar **mengering bertahap**. Hujan 1–3 bulan lalu masih mempengaruhi kerawanan hari ini. Maka hujan dimasukkan pada lag 0, 1, 2, 3 (suhu & kelembapan hanya bulan berjalan).

### Interpretasi koefisien (per 1 simpangan baku kenaikan):
- Bagian kejadian: **exp(β) = rasio odds** (OR). OR 0,63 → peluang (odds) turun 37%.
- Bagian jumlah: **exp(β) = rate ratio** (RR). RR 0,82 → jumlah turun 18%.

### Standardisasi tanpa "mengintip" data uji
Setiap kovariat dikurangi rerata dan dibagi SB **dari data latih saja** → mencegah *kebocoran data* (data leakage).
""")
code(r"""
# 1) hujan musiman & tahun kering
rain_mean_cell = d["rain"].mean(1)
fig, ax = plt.subplots(1, 3, figsize=(17, 3.8))
ax[0].plot(rain_mean_cell, c="#4c78a8"); ax[0].set_title("Hujan rata-rata per bulan (simulasi)\nmusim kering berulang + tahun 'El Niño' lebih kering")
ax[0].set_xlabel("bulan ke-"); ax[0].set_ylabel("mm/bulan")
# 2) lag: korelasi log total hotspot dengan hujan lag l (mirip Gambar 4.6)
ltot = np.log1p(Y.sum(1)); cors = []
for l in range(4): cors.append(np.corrcoef(ltot, d["X_raw"][f"hujan_lag{l}"].mean(1))[0,1])
ax[1].bar(range(4), cors, color="#e45756"); ax[1].set_xticks(range(4)); ax[1].set_xlabel("lag hujan (bulan)")
ax[1].set_title("Korelasi log hotspot agregat dengan hujan\n(tesis: −0,64, −0,68, −0,63, −0,35)")
# 3) pengaruh lag: contoh efek pada peluang
x = np.linspace(-2.5, 2.5, 100)
for l, bz in enumerate([-0.60, -0.30, -0.15, 0.0]):
    ax[2].plot(x, expit(0.15 + bz*x), label=f"lag {l} (β={bz})")
ax[2].set_title("Efek hujan lag‑l pada peluang kejadian\n(sel/bulan lain dijaga tetap)"); ax[2].set_xlabel("hujan terstandardisasi (SB)"); ax[2].set_ylabel("P(ada hotspot)"); ax[2].legend()
plt.tight_layout(); plt.show()
""")
md(r"""
### ⚠️ Peringatan multikolinearitas & pembaur spasial
- Hujan bulan-bulan berdekatan **saling berkorelasi**, juga suhu/kelembapan → koefisien individual tak stabil walau efek gabungannya tetap terukur.
- Tesis menemukan **tanda suhu berlawanan** antara dua bagian dan **tanda kelembapan berbalik** saat efek laten ditambahkan — sejalan dengan *spatial confounding* (Hodges & Reich, 2010). Kesimpulan tesis: **hanya pola hujan berlag yang layak ditafsirkan**.

Kita akan melihat gejala serupa di Bagian 11.
""")

# =============================================================================
md(r"""
---
# BAGIAN 6 — Komponen bersama (*copy*) antara kejadian & jumlah (kotak 6)

Dua bagian hurdle punya prediktor sendiri, tetapi **boleh berbagi pola laten yang sama** dengan **skala bebas b**:

> efek pada bagian jumlah = **b × efek pada bagian kejadian**

| Nilai b | Arti |
|---|---|
| b ≈ 1 | pola untuk *jumlah* sama amplitudonya dengan pola untuk *kejadian* |
| b > 1 | variasi itu **lebih kuat** pada jumlah |
| 0 < b < 1 | lebih lemah pada jumlah |

Temuan tesis: **b ruang ≈ 0,9 dan b musim ≈ 1,0** (sama), tetapi **b waktu ≈ 2** — tahun kering menggeser **jumlah** hotspot jauh lebih kuat daripada menggeser **peluang** kejadian (pada tahun kering sebagian besar sel sudah positif, jadi variasi tinggal di jumlah).
""")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8), sharey=True)
u_demo = tr["u"]
for a, b in zip(ax, [0.5, 1.0, 2.0]):
    a.plot(u_demo, label="efek waktu bagian kejadian (u_t)", c="#4c78a8")
    a.plot(b*u_demo, label=f"bagian jumlah = {b}×u_t", c="#e45756")
    a.set_title(f"b = {b}"); a.set_xlabel("bulan ke-")
ax[0].legend(fontsize=8); plt.suptitle("Fitur 'copy' INLA: pola SAMA, amplitudo diskalakan oleh b", y=1.03); plt.tight_layout(); plt.show()
""")
md(r"""
> ⚠️ **Catatan identifikasi (dari tesis):** b hanya bermakna *relatif* terhadap skala efek di bagian kejadian — tafsirkan sebagai **rasio amplitudo**, bukan besaran absolut.

### Interaksi ruang–waktu (ZH2, ZH3)
Struktur di atas **aditif**: pola ruang diasumsikan tetap sepanjang waktu. Interaksi (Knorr‑Held, 2000) memberi efek tambahan di setiap pasangan sel‑bulan:
- **Jenis I (ZH2):** efek independen (iid) per sel‑bulan — ribuan simpul laten tambahan (47.456 vs 992 di tesis).
- **Besag × AR(1) (ZH3):** dependensi ruang yang berkembang bertahap dalam waktu — paling berat; tidak dijalankan di tesis.

Temuan tesis: interaksi iid memperbaiki DIC (kecocokan dalam sampel) 1.681 poin, **tetapi tidak memperbaiki prediksi** (selisih CRPS 0,026, selang −0,092…0,165) dengan biaya **7,7× lebih lama**. *Kecocokan dalam sampel yang lebih baik ≠ prediksi yang lebih baik.*

---
# BAGIAN 7 — Inti INLA: aproksimasi Laplace (kotak 8)

### Masalahnya
Model bertingkat punya ribuan komponen laten (tesis: 992 simpul pada ZH1). Cara klasik (**MCMC**) mengambil jutaan sampel — lambat. **INLA** menghitung posterior secara **deterministik** lewat trik:
> Banyak distribusi posterior berbentuk mendekati **lonceng (Gauss)** → cukup cari **puncaknya (modus)** dan **kelengkungannya** di puncak.

Itulah **aproksimasi Laplace**. Mari lihat pada contoh mini satu parameter (laju λ dari data Poisson dengan prior lognormal):
""")
code(r"""
from scipy.optimize import minimize_scalar
yobs = np.array([3, 7, 2, 12, 5])
eta = np.linspace(-1, 4, 600)                       # eta = log(lambda)
logpost = lambda e: (yobs*e - np.exp(e)).sum() - 0.5*(e-1.0)**2/1.5**2
lp = np.array([logpost(e) for e in eta]); post = np.exp(lp-lp.max()); post /= np.trapezoid(post, eta)
m = minimize_scalar(lambda e: -logpost(e), bounds=(-1,4), method="bounded").x
h = 1e-4; curv = -(logpost(m+h) - 2*logpost(m) + logpost(m-h))/h**2   # -turunan kedua
lap = stats.norm.pdf(eta, m, 1/np.sqrt(curv))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].plot(eta, post, lw=3, c="#999", label="posterior EKSAK (hitung brute-force)")
ax[0].plot(eta, lap, "--", lw=2, c="#d62728", label="aproksimasi Laplace (Gauss di modus)")
ax[0].axvline(m, c="k", lw=.8); ax[0].legend(); ax[0].set_title("Laplace: cari puncak + kelengkungan"); ax[0].set_xlabel("η = log λ")
ax[1].plot(eta, np.log(post+1e-300), lw=3, c="#999"); ax[1].plot(eta, np.log(lap+1e-300), "--", c="#d62728", lw=2)
ax[1].set_ylim(-12, 1); ax[1].set_title("Skala log: sangat dekat di sekitar puncak"); ax[1].set_xlabel("η")
plt.tight_layout(); plt.show()
print(f"Modus η* = {m:.3f}; SB aproksimasi = {1/np.sqrt(curv):.3f}")
""")
md(r"""
### Apa yang INLA lakukan sebenarnya (tiga lapis)
1. **Lapis dalam** — untuk hiperparameter θ tertentu (σ, ρ, φ, k, …), cari modus semua efek laten **x*** dan kelengkungannya (Laplace). Hasilnya: Gauss multivariat untuk efek laten.
2. **Lapis tengah** — cari posterior hiperparameter π(θ|y) (juga dengan Laplace).
3. **Lapis luar** — gabungkan atas θ. **Tesis memakai *empirical Bayes*:** θ ditetapkan di modusnya (tak diintegralkan) agar 9 model × beberapa lipatan tetap cepat. **Konsekuensi** yang diakui tesis: ketidakpastian hiperparameter tidak masuk ke selang kredibel (cenderung terlalu sempit).

### Prior "penalized complexity" (PC prior)
Prior yang menyusutkan model menuju versi sederhana. Tesis: **P(σ > 1) = 0,01** untuk setiap komponen laten — "kecil kemungkinannya efek laten sangat besar kecuali data memaksa". Mari lihat bentuknya:
""")
code(r"""
sig = np.linspace(0.001, 3, 300)
lam = -np.log(0.01)/1.0                                   # P(σ>1)=0.01 → eksponensial
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(sig, lam*np.exp(-lam*sig), lw=2); ax.axvline(1, c="r", ls=":")
ax.fill_between(sig[sig>1], lam*np.exp(-lam*sig[sig>1]), color="r", alpha=.3, label="luas = 0,01 (P(σ>1))")
ax.set_title("PC prior untuk σ: menyusutkan ke 0 (model sederhana)"); ax.set_xlabel("σ"); ax.legend(); plt.show()
""")

# =============================================================================
md(r"""
---
# BAGIAN 8 — Membagi data & melatih model (kotak 4, 8)

### Pembagian data harus **menurut waktu**
Mengacak baris membuat data latih "mengintip masa depan". Tesis: latih bulan 1–93, uji bulan 94–132 (39 bulan). Kita meniru: **latih 72 bulan pertama, uji 36 bulan terakhir** (data simulasi sengaja memuat tahun kering di bagian uji).
""")
code(r"""
TRAIN_T = 72
fig, ax = plt.subplots(figsize=(12, 2.6))
ax.bar(range(TRAIN_T), Y.sum(1)[:TRAIN_T], color="#4c78a8", label="LATIH (untuk belajar)")
ax.bar(range(TRAIN_T, T), Y.sum(1)[TRAIN_T:], color="#e45756", label="UJI (disembunyikan)")
ax.set_yscale("log"); ax.legend(); ax.set_xlabel("bulan ke-"); ax.set_ylabel("total hotspot"); ax.set_title("Pembagian latih/uji menurut waktu")
plt.show()
""")
md(r"""
### Enam model yang dibandingkan (versi ringkas dari 9 model tesis)
| Kode | Isi | Tujuan perbandingan |
|---|---|---|
| **M0** | Poisson; laju sel = rerata historis sel | pembanding minimal |
| **M0s** | Poisson; laju = rerata historis sel **pada bulan kalender yang sama** | "klimatologi musiman" — lawan yang berat! |
| **GNB** | Regresi NB + iklim, tanpa efek laten | nilai tambah efek laten |
| **ZN** | NB + efek laten (BYM2, AR(1), musim) + iklim, **tanpa komponen nol** | **lawan utama hurdle** |
| **ZH0** | Hurdle + laten, **tanpa** iklim | nilai tambah iklim |
| **ZH1** | Hurdle + laten + iklim | **model utama tesis** |

*(GH, ZH2, ZH3 dihilangkan agar waktu jalan singkat; kodenya sudah ada di `hurdle_lab.py` — `LatentModel(d, 72, "GH")` dst.)*

**Prinsip adil:** panel, standardisasi, prior, dan pembagian data **dijaga sama** untuk semua model, sehingga perbedaan skor bisa dikaitkan pada komponen yang sengaja diubah.

### Melatih (±3–4 menit). Sambil menunggu, ingat: tiap model mencari modus posterior (L‑BFGS) lalu menghitung kelengkungan (Hessian) → Laplace.
""")
code(r"""
t0 = time.time()
models = {}
for kind, ksz in [("GNB", 0.3), ("ZN", 0.6), ("ZH0", 0.9), ("ZH1", 0.9)]:
    t = time.time()
    models[kind] = LatentModel(d, TRAIN_T, kind, k_size=ksz).fit()
    print(f"{kind:4s} selesai {time.time()-t:5.1f} dtk | iterasi {models[kind].res.nit:4d} | -log posterior {models[kind].res.fun:,.1f}")
print(f"Total: {time.time()-t0:.0f} detik")
""")
md(r"""
Baseline M0 dan M0s tidak butuh INLA — cukup rerata, lalu sampel dari distribusi Poisson:
""")
code(r"""
S = 300
te_mask = np.arange(T) >= TRAIN_T
te_idx = np.where(np.repeat(te_mask, n))[0]
y_te = Y.reshape(-1)[te_idx]
time_te = np.repeat(np.arange(T), n)[te_idx]; cell_te = np.tile(np.arange(n), T)[te_idx]
mon_te = d["month"][time_te]

# M0: laju sel = rerata sel pada data latih
m0_rate = Y[:TRAIN_T].mean(0)
# M0s: laju sel x bulan kalender
mon_tr = d["month"][:TRAIN_T]
m0s_rate = np.array([[Y[:TRAIN_T][mon_tr == m, c].mean() if (mon_tr==m).any() else m0_rate[c] for c in range(n)] for m in range(12)])
r = np.random.default_rng(3)
preds = {"M0": r.poisson(m0_rate[cell_te][:, None] * np.ones((1, S))),
         "M0s": r.poisson(m0s_rate[mon_te, cell_te][:, None] * np.ones((1, S)))}
pis = {}
for k, m in models.items():
    s = m.predict_samples(S=S, seed=7)
    assert (s["idx"] == te_idx).all()
    preds[k] = s["y"]; pis[k] = s["pi"]
order = ["M0", "M0s", "GNB", "ZN", "ZH0", "ZH1"]
print("Sampel prediktif siap:", {k: v.shape for k, v in preds.items()})
""")
md(r"""
### Cara prediksi dibuat untuk satu baris uji (kotak 9)
Untuk model hurdle, **300 kali** diulang:
1. Tarik η_kejadian dan η_jumlah dari Gauss (rerata & SB posterior dari Laplace).
2. π = logistik(η_kejadian); μ = exp(η_jumlah).
3. Dengan peluang **1−π** → hitungan **0**; dengan peluang **π** → hitungan **1 + NB(μ, k)**.

Hasilnya bukan satu angka, melainkan **sebaran prediksi**. Lihat satu contoh:
""")
code(r"""
i = int(np.argmax(y_te))  # baris uji terbesar
fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
for a, k in zip(ax, ["GNB", "ZN", "ZH1"]):
    a.hist(np.log1p(preds[k][i]), bins=40, color="#4c78a8", alpha=.85)
    a.axvline(np.log1p(y_te[i]), c="r", lw=2, label=f"teramati = {y_te[i]}"); a.set_title(f"{k}: sebaran prediksi (log1p)"); a.legend()
plt.suptitle(f"Baris uji dengan hitungan tertinggi (sel {cell_te[i]}, bulan ke-{time_te[i]+1})", y=1.03); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""
---
# BAGIAN 9 — Menilai prediksi (kotak 10–11)

## 9.1 CRPS: skor utama
**CRPS** menilai *seluruh sebaran prediksi* terhadap satu nilai nyata: kecil bila sebaran **akurat dan tajam**. Satuannya sama dengan data (hotspot). **Lebih kecil = lebih baik.** Skor ini "proper": tidak bisa dicurangi dengan melaporkan sebaran palsu.

Intuisi rumus: CRPS = *rata-rata jarak prediksi ke kenyataan* − *½ rata-rata jarak antar-prediksi*. Suku kedua menghadiahi sebaran yang tajam.
""")
code(r"""
# demonstrasi intuisi: tiga 'peramal' untuk hasil nyata = 20
r4 = np.random.default_rng(0)
cands = {"akurat & tajam": r4.normal(20, 3, 3000), "akurat tapi terlalu lebar": r4.normal(20, 25, 3000),
         "tajam tapi meleset": r4.normal(40, 3, 3000)}
fig, ax = plt.subplots(figsize=(8, 3.2))
for nm, s in cands.items():
    c = crps_samples(s[None, :300], np.array([20.]))[0]
    ax.hist(s, bins=60, alpha=.55, label=f"{nm}: CRPS={c:.1f}")
ax.axvline(20, c="k", lw=2); ax.legend(fontsize=8); ax.set_title("CRPS menghukum sebaran yang meleset ATAU terlalu lebar"); plt.show()
""")
code(r"""
rows = []
for k in order:
    sm = preds[k]; med = np.median(sm, 1); mean = sm.mean(1)
    lo, hi = np.percentile(sm, [5, 95], 1); lo5, hi5 = np.percentile(sm, [25, 75], 1)
    rows.append(dict(Model=k, CRPS=crps_samples(sm, y_te).mean(),
                     MAE=np.abs(med - y_te).mean(), RMSE=np.sqrt(((mean - y_te)**2).mean()),
                     **{"Cakupan 90% (%)": ((y_te >= lo) & (y_te <= hi)).mean()*100,
                        "Cakupan 50% (%)": ((y_te >= lo5) & (y_te <= hi5)).mean()*100,
                        "Bias total (%)": (mean.sum()/y_te.sum() - 1)*100}))
tab = pd.DataFrame(rows).set_index("Model")
tab["Skill CRPS vs M0 (%)"] = (1 - tab.CRPS/tab.loc["M0","CRPS"])*100
display(tab.round(2))
""")
md(r"""
**Cara membaca (padanan Tabel 4.5 tesis):**
- **CRPS/MAE/RMSE**: kecil lebih baik. **RMSE** sangat peka pada beberapa nilai ekstrem (kuadrat) — di tesis, ZN tampak jelek di RMSE karena melebihkan satu puncak.
- **Cakupan 90%** idealnya ≈ 90; **50%** ≈ 50. Jauh di bawah → terlalu sempit (sok yakin); jauh di atas → terlalu lebar.
- **Bias total**: selisih relatif total prediksi vs total teramati.
- **Skill** = seberapa persen CRPS turun dibanding M0.
""")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
colors = ["#bbb","#999","#f58518","#e45756","#72b7b2","#4c78a8"]
ax[0].barh(order, tab.CRPS, color=colors); ax[0].set_title("CRPS (kecil = baik)"); ax[0].invert_yaxis()
ax[1].barh(order, tab["Skill CRPS vs M0 (%)"], color=colors); ax[1].set_title("Skill terhadap M0 (%)"); ax[1].invert_yaxis()
ax[2].scatter(tab["Cakupan 90% (%)"], tab["Cakupan 50% (%)"], c=colors, s=120, zorder=3)
for k in order: ax[2].annotate(k, (tab.loc[k,"Cakupan 90% (%)"]+.8, tab.loc[k,"Cakupan 50% (%)"]+.8))
ax[2].plot(90, 50, "k*", ms=16, label="ideal (nominal)"); ax[2].set_xlabel("cakupan 90%"); ax[2].set_ylabel("cakupan 50%"); ax[2].legend(); ax[2].set_title("Kalibrasi interval")
plt.tight_layout(); plt.show()
""")
md(r"""
## 9.2 PIT: apakah sebaran prediksi "jujur"?
Untuk setiap baris uji, hitung posisi nilai nyata di dalam sebaran prediksinya (PIT). Jika model **terkalibrasi**, semua posisi itu **tersebar merata** → histogram **datar**.
- **Bentuk U** → model terlalu sempit/sok yakin. **Bentuk punuk** → terlalu lebar. **Miring** → bias.

Untuk data cacah, PIT diacak sedikit (Czado dkk., 2009) agar bisa seragam.
""")
code(r"""
rp = np.random.default_rng(4)
fig, ax = plt.subplots(1, 6, figsize=(18, 2.8), sharey=True)
for a, k in zip(ax, order):
    p = pit_randomized(preds[k], y_te, rp)
    a.hist(p, bins=10, range=(0,1), density=True, color="#4c78a8"); a.axhline(1, c="r", ls="--"); a.set_title(k); a.set_xlabel("PIT")
plt.suptitle("Histogram PIT (garis merah = ideal datar)", y=1.05); plt.show()
""")
md(r"""
## 9.3 Skor khusus **kejadian**: peluang ada hotspot
Karena komponen kejadian adalah pokok penelitian, kita nilai peluang **P(Y ≥ 1)** langsung:
- **AUC** — kemampuan *membedakan* sel‑bulan yang menyala vs tidak (1 = sempurna, 0,5 = tebakan).
- **Brier** — galat kuadrat peluang (kecil = baik).
- **Diagram reliabilitas** — jika model berkata "70%", apakah memang ≈70% kejadian muncul? Titik harus menempel diagonal.
- **ECE** — rata-rata jarak titik ke diagonal (kecil = baik).
""")
code(r"""
from sklearn.metrics import roc_auc_score
ev = (y_te >= 1).astype(float)
rows = []; fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
for k, c in zip(order, colors):
    p = (preds[k] >= 1).mean(1)
    rows.append(dict(Model=k, AUC=roc_auc_score(ev, p + 1e-9*np.arange(len(p))), Brier=((p-ev)**2).mean(), ECE=ece(p, ev)))
    bins = np.linspace(0, 1, 11); b = np.clip(np.digitize(p, bins)-1, 0, 9)
    xs = [p[b==i].mean() for i in range(10) if (b==i).sum() > 5]; ys_ = [ev[b==i].mean() for i in range(10) if (b==i).sum() > 5]
    ax[0].plot(xs, ys_, "o-", c=c, label=k, ms=5)
ax[0].plot([0,1],[0,1],"k--"); ax[0].set_xlabel("peluang prediksi"); ax[0].set_ylabel("frekuensi teramati"); ax[0].legend(); ax[0].set_title("Diagram reliabilitas P(Y≥1)")
kt = pd.DataFrame(rows).set_index("Model"); display(kt.round(3))
ax[1].barh(order, kt.ECE, color=colors); ax[1].invert_yaxis(); ax[1].set_title("ECE (kecil = peluang kejadian lebih jujur)")
plt.tight_layout(); plt.show()
""")
md(r"""
> **Pola yang diharapkan dari tesis:** pada AUC/Brier hurdle (ZH1) dan NB laten (ZN) hampir setara, tetapi hurdle unggul pada **kalibrasi** (ECE 0,026 vs 0,060 di tesis). Cek apakah data simulasi kita menunjukkan hal serupa.

## 9.4 Proporsi nol & total bulanan: apakah model menangkap musimnya?
""")
code(r"""
mon_idx = np.arange(TRAIN_T, T)
obs_zero = np.array([(Y[t] == 0).mean() for t in mon_idx]); obs_tot = Y[TRAIN_T:].sum(1)
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax[0].plot(mon_idx, obs_zero, "k--", lw=2.2, label="teramati")
ax[1].plot(mon_idx, obs_tot, "k-", lw=2.4, label="teramati")
for k, c in zip(["M0s","GNB","ZN","ZH1"], ["#999","#f58518","#e45756","#4c78a8"]):
    sm = preds[k].reshape(n, -1) if False else preds[k]
    zs, ts = [], []
    for j, t in enumerate(mon_idx):
        rows_t = np.where(time_te == t)[0]
        zs.append((sm[rows_t] == 0).mean()); ts.append(sm[rows_t].mean(1).sum())
    ax[0].plot(mon_idx, zs, c=c, label=k); ax[1].plot(mon_idx, ts, c=c, label=k)
ax[0].set_title("Proporsi sel bernilai nol per bulan uji"); ax[0].legend(ncol=5)
ax[1].set_yscale("log"); ax[1].set_title("Total hotspot per bulan uji (skala log)"); ax[1].set_xlabel("bulan ke-")
plt.tight_layout(); plt.show()
""")
md(r"""
## 9.5 *Di mana* komponen nol membantu? — analisis subset (kotak 11)
Rerata menyeluruh bisa **menyamarkan** kemenangan yang terkonsentrasi. Tesis membelah data uji menjadi: **musim api** (Jul–Okt), **bukan musim**, **bulan ekstrem** (10% bulan dengan total tertinggi), dan lainnya.
""")
code(r"""
crps_row = {k: crps_samples(preds[k], y_te) for k in order}
tot_te = np.array([Y[t].sum() for t in range(T)])
ext_months = set(mon_idx[np.argsort(tot_te[TRAIN_T:])[::-1][:max(1, int(round(.1*len(mon_idx))))]])
is_fire = np.isin(mon_te, [6, 7, 8, 9])  # Jul..Okt (indeks 6..9)
is_ext = np.isin(time_te, list(ext_months))
subsets = {"Semua": np.ones(len(y_te), bool), "Musim api (Jul–Okt)": is_fire, "Bukan musim": ~is_fire, "Bulan ekstrem": is_ext, "Bulan lain": ~is_ext}
st = pd.DataFrame({nm: {k: crps_row[k][m].mean() for k in order} for nm, m in subsets.items()})
st.loc["ZH1 lebih baik dari ZN (%)"] = (1 - st.loc["ZH1"]/st.loc["ZN"])*100
display(st.round(2))
sub = st.drop("ZH1 lebih baik dari ZN (%)").drop(["M0","M0s"])
ax = sub.T.plot(kind="bar", figsize=(12, 3.8), color=["#f58518","#e45756","#72b7b2","#4c78a8"]); ax.set_ylabel("CRPS"); ax.set_title("CRPS menurut subset (kecil = baik)"); plt.xticks(rotation=0); plt.show()
print("Tesis: ZH1 vs ZN lebih rendah 8,8% pada musim api dan 11,4% pada bulan ekstrem; pada bulan bukan musim praktis sama.")
""")

# =============================================================================
md(r"""
---
# BAGIAN 10 — Apakah selisih antar model *nyata*? (kotak 11)

Selisih rata-rata CRPS yang kecil bisa hanya kebetulan. Tesis memakai:
1. **Selisih berpasangan per bulan**: d_t = CRPS_acuan,t − CRPS_model,t (positif → model lebih baik pada bulan t).
2. **Bootstrap blok** (blok 3 bulan): pengacakan ulang *potongan berurutan* agar ketergantungan antarbulan dipertahankan → selang kepercayaan 95%.

**Aturan baca:** jika selang **tidak memuat 0**, selisih dianggap meyakinkan.
""")
code(r"""
def monthly_crps(k):
    return np.array([crps_row[k][time_te == t].mean() for t in mon_idx])
mc = {k: monthly_crps(k) for k in order}
pairs = [("ZN", "GNB", "efek laten"), ("ZH1", "ZN", "komponen nol"), ("ZH1", "ZH0", "iklim"), ("ZH1", "M0s", "vs klimatologi musiman")]
out = []
fig, ax = plt.subplots(figsize=(10, 3.6))
for j, (a, b, ket) in enumerate(pairs):
    dlt = mc[b] - mc[a]           # b = acuan; positif → a lebih baik
    lo, hi = block_bootstrap_ci(dlt, B=1000, block=3, seed=j)
    out.append(dict(Perbandingan=f"{a} vs {b} ({ket})", dCRPS_rerata=dlt.mean(), CI_bawah=lo, CI_atas=hi,
                    **{"% bulan A lebih baik": (dlt > 0).mean()*100}))
    ax.errorbar(dlt.mean(), j, xerr=[[dlt.mean()-lo],[hi-dlt.mean()]], fmt="o", c="#4c78a8", capsize=4)
ax.axvline(0, c="k"); ax.set_yticks(range(len(pairs))); ax.set_yticklabels([f"{a} vs {b}\n({k})" for a,b,k in pairs]); ax.invert_yaxis()
ax.set_xlabel("selisih CRPS (positif = model A lebih baik); bar = selang 95% bootstrap blok"); plt.tight_layout(); plt.show()
display(pd.DataFrame(out).set_index("Perbandingan").round(2))
""")
md(r"""
> **Pelajaran dari tesis:** ZH1 vs ZN hanya selisih ≈1,23 CRPS dengan batas bawah selang 0,11 — *"nyata tetapi kecil"*; di 2 dari 3 lipatan rolling-origin selangnya memuat 0. Dan M0s (klimatologi musiman) **tidak terkalahkan** secara signifikan pada tahun uji 2023. Selalu laporkan pembanding sederhana yang kuat.

### Stabilitas: *rolling-origin*
Satu pembagian bisa kebetulan. Tesis mengulang dengan beberapa titik asal (F1, F2, F3). Berikut versi ringkas pada data kita: latih sampai bulan 48, 60, 72 → uji 12 bulan berikutnya (hanya GNB, ZN, ZH1 agar cepat; tiap pengulangan ±1–2 menit).
""")
code(r"""
t0 = time.time(); roll = []
for tr_end in [48, 60]:
    row = {"asal (latih s.d. bulan)": tr_end}
    te_slice = np.arange(tr_end, tr_end + 12)
    for kind, ksz in [("GNB", .3), ("ZN", .6), ("ZH1", .9)]:
        m = LatentModel(d, tr_end, kind, k_size=ksz).fit()
        s = m.predict_samples(S=200, seed=2)
        idx = s["idx"]; sel = np.isin(m.time[idx], te_slice)
        row[kind] = crps_samples(s["y"][sel], m.y[idx][sel]).mean()
    roll.append(row)
roll.append({"asal (latih s.d. bulan)": 72, **{k: tab.loc[k, "CRPS"] for k in ["GNB","ZN","ZH1"]}})
rollt = pd.DataFrame(roll).set_index("asal (latih s.d. bulan)")
display(rollt.round(2)); print(f"({time.time()-t0:.0f} dtk) — baris 72 = split utama (uji 36 bln), dua baris lain = uji 12 bln")
""")

# =============================================================================
md(r"""
---
# BAGIAN 11 — Membaca hasil: apa yang model temukan? (kotak 12)

Karena ini data simulasi, kita punya kelebihan yang tidak dimiliki tesis: **kita bisa mengecek temuan model terhadap kebenaran**.

## 11.1 Pola ruang (padanan Gambar 4.14)
""")
code(r"""
mz = models["ZH1"]; eff = mz.effects()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for a, v, tt in zip(ax[:2], [tr["a"], eff["a"]], ["Efek ruang BENAR (simulasi)", "Efek ruang DIESTIMASI (ZH1)"]):
    s = a.scatter(xy[:,0], xy[:,1], c=v, s=700, marker="s", cmap="RdBu_r", vmin=-2.5, vmax=2.5); a.set_title(tt)
    a.set_aspect("equal"); a.grid(False); a.set_xticks([]); a.set_yticks([])
plt.colorbar(s, ax=ax[:2], shrink=.8)
ax[2].scatter(tr["a"], eff["a"], c="#4c78a8"); lim = [-3, 3]; ax[2].plot(lim, lim, "k--")
ax[2].set_xlabel("benar"); ax[2].set_ylabel("estimasi"); ax[2].set_title(f"korelasi = {np.corrcoef(tr['a'], eff['a'])[0,1]:.3f}")
plt.show()
""")
md(r"""
Peta "efek ruang" = **kerawanan menetap** yang tidak dijelaskan oleh iklim dan musim (di data asli: kemungkinan gambut, tutupan lahan, akses). Tesis menekankan: tafsirkan sebagai pola, **bukan bukti sebab**.

## 11.2 Pola musim & waktu (padanan Gambar 4.15–4.16)
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].plot(mn, tr["season"] - tr["season"].mean(), "k-o", label="benar (bagian kejadian)")
ax[0].plot(mn, eff["s"] - eff["s"].mean(), "r--o", label="estimasi (bagian kejadian)")
ax[0].plot(mn, eff["b"][2]*(eff["s"] - eff["s"].mean()), ":s", c="#4c78a8", label=f"estimasi bagian jumlah (b={eff['b'][2]:.2f})")
ax[0].legend(); ax[0].set_title("Profil musim")
ax[1].plot(tr["u"], "k-", label="benar"); ax[1].plot(eff["u"][:T], "r--", label="estimasi")
ax[1].axvline(TRAIN_T-.5, c="gray", ls=":"); ax[1].text(TRAIN_T+1, ax[1].get_ylim()[1]*.8, "← latih | uji →", fontsize=9)
ax[1].legend(); ax[1].set_title("Efek waktu AR(1) — di bulan uji, estimasi 'kembali ke 0' karena tak ada data respons")
plt.tight_layout(); plt.show()
""")
md(r"""
> Perhatikan area uji di panel kanan: karena model belum "melihat" respons bulan uji, efek AR(1)-nya **meluruh ke nol**. Di sinilah **iklim** harus bekerja menjelaskan tahun kering/basah — dan kenapa tesis menemukan iklim **menambah nilai prediktif** (CRPS −1,41). Tesis juga menyarankan menambah indeks ENSO/DMI karena sisa temporal masih memuncak di 2015, 2019, 2023.

## 11.3 Koefisien skala *copy* b
""")
code(r"""
bt = tr["b"]
print("          estimasi   benar (simulasi)     |   tesis")
for nm, i, t_, th in [("b ruang", 0, bt["a"], "0,911 (0,848–0,977)"), ("b waktu", 1, bt["u"], "2,035 (1,779–2,295)"), ("b musim", 2, bt["s"], "1,011")]:
    print(f"{nm:9s} {eff['b'][i]:8.2f}   {t_:8.2f}   {'':8s}| {th}")
""")
md(r"""
## 11.4 Efek iklim: rasio odds & rate ratio (padanan Gambar 4.13 / Tabel 4.15)
Kita bandingkan estimasi **ZH1** (dengan efek laten) dengan **kebenaran** simulasi. Selang kredibel dihitung dari Hessian (Laplace).
""")
code(r"""
names = ["intersep"] + d["names"]
pf = 1 + mz.p
def ci_from(model, i):
    sd = np.sqrt(model.Sigma[i, i]); return model.x[i] - 1.96*sd, model.x[i] + 1.96*sd
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
for a, part, off, tdict, ttl, lab in [(ax[0], "kejadian", 0, tr["beta_z"], "Bagian KEJADIAN — rasio odds", "OR"),
                                      (ax[1], "jumlah", pf, tr["beta_n"], "Bagian JUMLAH — rate ratio", "RR")]:
    for j, nm in enumerate(d["names"], start=1):
        lo, hi = ci_from(mz, off + j); est = mz.x[off + j]
        a.errorbar(np.exp(est), j, xerr=[[np.exp(est)-np.exp(lo)],[np.exp(hi)-np.exp(est)]], fmt="o", c="#4c78a8", capsize=3, label="estimasi ZH1 (95%)" if j==1 else None)
        a.plot(np.exp(tdict[nm]), j, "r*", ms=12, label="kebenaran" if j==1 else None)
    a.axvline(1, c="k"); a.set_yticks(range(1, len(d["names"])+1)); a.set_yticklabels(d["names"]); a.invert_yaxis(); a.set_title(ttl); a.set_xlabel(f"{lab} per +1 SB"); a.legend()
plt.tight_layout(); plt.show()
""")
md(r"""
### Gejala "spatial confounding" — dengan atau tanpa efek laten?
Bandingkan koefisien iklim pada **GNB (tanpa laten)** dan **ZN (dengan laten)** (keduanya bagian jumlah). Jika kovariat terstruktur dalam ruang-waktu, koefisien bisa **bergeser** saat efek laten ditambahkan (Hodges & Reich, 2010). Di tesis: tanda kelembapan *berbalik* antara GH dan ZH1.
""")
code(r"""
cmp = pd.DataFrame({"benar (RR jumlah)": [np.exp(tr["beta_n"][k]) for k in d["names"]],
                    "GNB (tanpa laten)": [np.exp(models["GNB"].x[j]) for j in range(1, pf)],
                    "ZN (dengan laten)": [np.exp(models["ZN"].x[j]) for j in range(1, pf)],
                    "ZH1 bag. jumlah": [np.exp(mz.x[pf + j]) for j in range(1, pf)]}, index=d["names"])
display(cmp.round(3))
cmp.plot(kind="bar", figsize=(11, 3.6), color=["k","#f58518","#e45756","#4c78a8"]); plt.axhline(1, c="gray", lw=.8)
plt.ylabel("rate ratio per +1 SB"); plt.title("Koefisien berubah tergantung spesifikasi model → hati-hati menafsirkan"); plt.xticks(rotation=0); plt.show()
""")

# =============================================================================
md(r"""
---
# BAGIAN 12 — Rangkuman, kesalahan umum, latihan, dan kode R-INLA asli

## 12.1 Rangkuman satu halaman
| Pertanyaan | Jawaban tesis |
|---|---|
| Masalah data? | 39,94% nol musiman + ekor sangat berat. Bukan "pencilan" — itu sinyal. |
| Model? | **Hurdle**: bagian 1 = *ada deteksi?* (logistik); bagian 2 = *berapa?* (NB geser). |
| Struktur? | Ruang **BYM2** (graf queen) + waktu **AR(1)** + musim **RW1 siklik**, dibagi dua bagian lewat **copy** (skala b). |
| Iklim? | Hujan lag 0–3 + suhu + kelembapan, koefisien terpisah di kedua bagian. |
| Cara estimasi? | **INLA** (Laplace bersarang), *empirical Bayes*, PC prior. |
| Cara menilai? | Data uji menurut waktu + rolling-origin; CRPS/LPD, PIT, ECE, subset; selisih berpasangan + bootstrap blok. |
| Hasil utama? | Hurdle > baseline kuat; atas NB laten: **kecil**, terkonsentrasi di **bulan api/ekstrem** dan **kalibrasi kejadian**; hujan berlag layak ditafsirkan; suhu/RH tidak; interaksi iid tidak sebanding biayanya. |
| Batasan? | Bukan prakiraan (iklim uji = teramati); resolusi 60 km; hotspot ≠ luas terbakar; prediksi independen antar-baris; nol = deteksi, bukan "tidak ada api". |

## 12.2 Kesalahan umum yang dihindari tesis
1. ❌ Menghapus nilai besar sebagai pencilan → ✅ memodelkannya sebagai ekor sah.
2. ❌ Membandingkan hurdle hanya dengan Poisson → ✅ dengan **NB laten setara (ZN)** yang adil.
3. ❌ Memilih model berdasar DIC/WAIC dalam sampel → ✅ skor **luar sampel** (DIC antar likelihood berbeda tak sebanding).
4. ❌ Acak latih/uji → ✅ pisah menurut **waktu**; standardisasi pakai statistik latih.
5. ❌ Mengandalkan satu split → ✅ rolling-origin + selisih berpasangan dengan selang.
6. ❌ Menafsirkan semua koefisien sebagai sebab → ✅ hanya pola yang stabil; waspada kolinearitas & spatial confounding.
7. ❌ Menyembunyikan hasil yang kurang menguntungkan → ✅ melaporkan bahwa M0s sulit dikalahkan & RMSE hurdle lebih buruk.

## 12.3 🧪 Latihan mandiri
1. **Nol vs ekor:** di `simulate()` ubah `muz` (intersep kejadian) dan `k_true`; pantau proporsi nol dan rasio ragam/rerata. Kapan NB tunggal cukup?
2. **Nilai tambah hurdle:** buat bagian kejadian & jumlah *dikendalikan hal yang sangat berbeda* (ubah `beta_z` vs `beta_n`). Apakah selisih ZH1–ZN membesar?
3. **Hiperparameter:** panggil `LatentModel(d, 72, "ZH1", rho=0.2)` atau `phi=0.2` dan bandingkan CRPS. Seberapa sensitif?
4. **Kebocoran data:** standardisasi memakai SELURUH data (bukan hanya latih) — apa yang berubah pada skor uji?
5. **Ukuran sampel prediktif:** ubah `S` dari 300 ke 30. Bagaimana CRPS dan histogram PIT?
6. **Tahun kering di uji:** ubah `dry_years` pada `simulate()` sehingga data uji tak memuat tahun kering. Apakah M0s lalu mengalahkan model laten (seperti pola tesis pada F2)?

## 12.4 Kode R-INLA yang sebenarnya (tidak dijalankan di sini)
Berikut kerangka kode R yang setara dengan ZH1 di tesis (disederhanakan). Jalankan di R dengan paket `INLA` terpasang.
""")
md(r"""
```r
library(INLA); library(spdep)

# --- 1. graf queen
nb  <- poly2nb(grid_sf, queen = TRUE);  nb2INLA("queen.graph", nb)

# --- 2. respons dua-likelihood: Y = list(z biner, n = y-1 pada y>0)
n_all <- nrow(panel)
Y <- list(matrix(c(panel$z, rep(NA, n_all)), ncol = 2),     # bagian kejadian (binomial)
          matrix(c(rep(NA, n_all), panel$y_shift), ncol = 2))  # bagian jumlah (NB)
# baris UJI diberi NA agar INLA menghasilkan prediktor linear untuk baris tsb.

# --- 3. indeks efek laten; copy memakai indeks NA pada bagian yang tak dipakai
pc_sig <- list(prior = "pc.prec", param = c(1, 0.01))        # P(sigma > 1) = 0.01
formula <- Y ~ -1 + mu_z + mu_n + rain0_z + rain1_z + ... + rain0_n + ... +
  f(cell_z, model = "bym2", graph = "queen.graph", scale.model = TRUE, constr = TRUE,
    hyper = list(prec = pc_sig, phi = list(prior = "pc", param = c(0.5, 2/3)))) +
  f(cell_n, copy = "cell_z", fixed = FALSE) +                  # skala b_a bebas
  f(time_z, model = "ar1",   hyper = list(prec = pc_sig)) +
  f(time_n, copy = "time_z", fixed = FALSE) +                  # b_u
  f(mon_z,  model = "rw1", cyclic = TRUE, scale.model = TRUE, hyper = list(prec = pc_sig)) +
  f(mon_n,  copy = "mon_z",  fixed = FALSE)                    # b_s

fit <- inla(formula, family = c("binomial", "nbinomial"), data = dat, E = area_offset,
            control.compute = list(dic = TRUE, waic = TRUE),
            control.inla = list(int.strategy = "eb"),          # empirical Bayes
            control.predictor = list(compute = TRUE))
# --- 4. sampel prediktif: S=300 tarik eta ~ N(mean, sd) per baris uji, lalu
#        y = 0 (peluang 1-pi) atau 1 + NB(mu, size)   ->  CRPS, PIT, ECE, ...
```

## 12.5 Bacaan lanjutan (dari daftar pustaka tesis)
- **INLA:** Rue, Martino & Chopin (2009); Rue dkk. (2017).
- **BYM2 & PC prior:** Riebler dkk. (2016); Simpson dkk. (2017).
- **Hurdle / zero-inflated:** Mullahy (1986); Lambert (1992).
- **Komponen bersama & interaksi:** Knorr-Held & Best (2001); Knorr-Held (2000).
- **Skor probabilistik:** Gneiting & Raftery (2007); Czado dkk. (2009).
- **Spatial confounding:** Hodges & Reich (2010). **DLNM:** Gasparrini dkk. (2010).

---
### Glosarium super singkat
| Istilah | Artinya |
|---|---|
| **Hotspot** | Piksel satelit yang terdeteksi sebagai api aktif (indikator, bukan luas terbakar) |
| **Overdispersi** | Data lebih "menyebar" daripada yang diizinkan Poisson |
| **Laten** | Tersembunyi — tak diukur langsung tapi diestimasi dari pola data |
| **Posterior** | Keyakinan tentang parameter *setelah* melihat data |
| **Prior** | Keyakinan awal sebelum melihat data |
| **Laplace** | Menghampiri bentuk posterior dengan lonceng Gauss di puncaknya |
| **Empirical Bayes** | Hiperparameter ditetapkan pada nilai terbaiknya, tidak dirata-ratakan |
| **CRPS** | Skor kualitas seluruh sebaran prediksi (kecil = baik) |
| **PIT / ECE** | Pemeriksaan kejujuran sebaran / peluang prediksi |
| **Rolling-origin** | Mengulang latih-uji dengan titik waktu berbeda untuk menguji kestabilan |
| **Bootstrap blok** | Pengacakan ulang potongan berurutan untuk menaksir ketidakpastian pada data berurutan |
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "Hands_On_Model_Hurdle_Kalimantan.ipynb")
print("ok", len(C), "sel")
