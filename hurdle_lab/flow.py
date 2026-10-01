"""Bagan alur data (dari data mentah sampai hasil akhir) - angka mengikuti tesis."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COL = dict(data="#DCEBFA", proc="#FFF1CC", model="#E3F4E1", eval="#F9DDE3", out="#E8E0F5", edge="#44546A")


def box(ax, x, y, w, h, title, body="", kind="proc", fs=10.6, num=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       fc=COL[kind], ec=COL["edge"], lw=1.2)
    ax.add_patch(p)
    head = (f"{num}  " if num else "") + title
    ax.text(x + w / 2, y + h - 0.16, head, ha="center", va="top", fontsize=fs + 1.2, fontweight="bold", color="#1F2D3D")
    if body:
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=fs, color="#2B2B2B", linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, a, b, side_a="b", side_b="t", txt=None, rad=0.0, color="#44546A"):
    def pt(bx, s):
        x, y, w, h = bx
        return {"t": (x + w / 2, y + h), "b": (x + w / 2, y), "l": (x, y + h / 2), "r": (x + w, y + h / 2)}[s]
    pa, pb = pt(a, side_a), pt(b, side_b)
    ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((pa[0] + pb[0]) / 2 + 0.1, (pa[1] + pb[1]) / 2, txt, fontsize=7.5, color=color, style="italic")


def draw_flow(path=None):
    fig, ax = plt.subplots(figsize=(19, 27))
    ax.set_xlim(0, 17); ax.set_ylim(0, 24); ax.axis("off")
    ax.text(8.5, 23.6, "ALUR DATA LENGKAP: dari titik api di satelit sampai hasil akhir",
            ha="center", fontsize=17, fontweight="bold", color="#1F2D3D")
    ax.text(8.5, 23.2, "(angka dalam kotak = angka nyata dari tesis; warna: biru=data, kuning=olah data, hijau=model, merah=evaluasi, ungu=hasil)",
            ha="center", fontsize=11, color="#555")

    # ---------- Lapis 1: data mentah -------------------------------------------------
    A = box(ax, 0.4, 20.3, 5.6, 2.4, "NASA FIRMS (VIIRS 375 m)",
            "Setiap deteksi api = 1 baris:\nkoordinat, tanggal, FRP, kepercayaan, tipe\n1.035.515 baris (2015-2026)", "data", num="1a")
    B = box(ax, 6.7, 20.3, 4.3, 2.4, "Batas provinsi",
            "Natural Earth: Kalbar, Kalteng,\nKalsel, Kaltim\n(Kaltara ikut Kaltim)", "data", num="1b")
    C = box(ax, 11.7, 20.3, 4.9, 2.4, "NASA POWER (MERRA-2)",
            "Iklim HARIAN di 396 titik:\nhujan, suhu 2 m, kelembapan\n396 x 4.261 hari = 1.687.356 baris", "data", num="1c")

    # ---------- Lapis 2: pembersihan -----------------------------------------------
    D = box(ax, 0.4, 17.3, 5.6, 2.3, "Saring hotspot",
            "Buang celah/NRT di luar Apr-2015..Mar-2026\nkepercayaan nominal/tinggi saja\ntipe vegetasi saja\n1.035.515 > 931.844 > 894.677 > 859.173", "proc", num="2a")
    E = box(ax, 6.7, 17.3, 4.3, 2.3, "Potong ke batas provinsi\nlalu beri sel grid 60 km",
            "Buang Malaysia/Brunei/laut\nSel daratan < 10% dibuang\n=> 176 sel (Kaltim 65, Kalbar 50,\nKalteng 48, Kalsel 13)", "proc", num="2b")
    F = box(ax, 11.7, 17.3, 4.9, 2.3, "Jadikan BULANAN lalu beri LAG",
            "hujan: dijumlah per bulan; suhu & RH: dirata-rata\nhujan lag 0,1,2,3 bulan sebelumnya\nPasangkan ke sel lewat titik iklim terdekat\n(median 21,5 km)", "proc", num="2c")

    # ---------- Lapis 3: panel -----------------------------------------------------
    G = box(ax, 0.4, 14.4, 5.6, 2.3, "PANEL SEL-BULAN",
            "Hitung deteksi per (sel c, bulan t)\n176 sel x 132 bulan = 23.232 baris\ntotal 826.771 hotspot\n39,94% baris bernilai NOL", "proc", num="3")
    H = box(ax, 6.7, 14.4, 4.3, 2.3, "Graf ketetanggaan QUEEN",
            "Sel yang bersentuhan sisi/sudut\n= tetangga (rerata 6,95 tetangga)\nuntuk efek ruang", "proc", num="3b")
    I = box(ax, 11.7, 14.4, 4.9, 2.3, "Bagi data SECARA WAKTU",
            "Latih: bln 1-93 (Apr-2015..Des-2022)\nUji  : bln 94-132 (Jan-2023..Mar-2026)\nStandardisasi kovariat HANYA dgn\nrerata/SB data latih (anti-bocor)", "proc", num="4")

    # ---------- Lapis 4: respons dipecah ------------------------------------------
    J = box(ax, 0.4, 11.5, 5.6, 2.3, "Pecah respons jadi DUA bagian",
            "Bagian 1 (kejadian): Z = 1 bila y > 0, else 0\nBagian 2 (jumlah)  : y-1 hanya utk baris y > 0\n(NB 'geser' sbg aproksimasi NB terpotong)", "model", num="5")
    K = box(ax, 6.7, 11.5, 4.3, 2.3, "Prediktor linear tiap bagian",
            "eta = intersep + iklim (hujan lag0-3,\nsuhu, RH) + ruang a_c + waktu u_t\n+ musim s_m(t)\nBagian 2 memakai SKALA copy b", "model", num="6")
    L = box(ax, 11.7, 11.5, 4.9, 2.3, "Prior (PC prior)",
            "BYM2: P(phi<0,5)=2/3 ; P(sigma>1)=0,01\nAR(1) & RW1 siklik: P(sigma>1)=0,01\nEfek tetap: Gauss presisi 0,001", "model", num="7")

    # ---------- Lapis 5: INLA ------------------------------------------------------
    M = box(ax, 3.2, 8.4, 10.6, 2.6, "INLA (aproksimasi Laplace bersarang) - estimasi 9 model",
            "M0, M0s (baseline Poisson) | GNB, GH (regresi tanpa laten) | ZN (NB laten) | ZH0, ZH1, ZH2, ZH3 (hurdle laten)\n"
            "Hiperparameter: empirical Bayes (ditetapkan di modus)   ->   posterior efek laten & koefisien (rerata, SB, selang 95%)\n"
            "Hanya baris LATIH berisi respons; baris UJI diberi NA agar INLA mengeluarkan prediktor linear-nya", "model", fs=10.6, num="8")

    # ---------- Lapis 6: prediksi --------------------------------------------------
    N = box(ax, 0.4, 5.5, 5.6, 2.3, "Sampel prediktif (S = 300)",
            "Untuk tiap baris uji: tarik eta ~ Normal(rerata, SB)\npi = logistik(eta1) ; mu = exp(eta2)\nhitungan = 0 (peluang 1-pi)\natau 1 + NB(mu, k) (peluang pi)", "eval", num="9")
    O = box(ax, 6.7, 5.5, 4.3, 2.3, "Skor luar sampel",
            "CRPS (utama), LPD, MAE, RMSE\ncakupan 90%/50%, PIT\nBrier, AUC, reliabilitas (ECE)\nproporsi nol & total bulanan", "eval", num="10")
    P = box(ax, 11.7, 5.5, 4.9, 2.3, "Subset & lipatan",
            "Musim api (Jul-Okt), bulan ekstrem,\nprovinsi; rolling-origin F1,F2,F3\nSelisih berpasangan per bulan +\nbootstrap blok (B=1000, blok 3)", "eval", num="11")

    # ---------- Lapis 7: hasil -----------------------------------------------------
    Q_ = box(ax, 0.4, 2.2, 5.6, 2.6, "HASIL 1: siapa yang menang?",
             "ZH1/ZH2 CRPS 15,0 (skill 65,4% atas M0)\nZH1 vs ZN: selisih kecil 1,23 (0,11-3,25)\nNilai tambah hurdle: musim api -8,8%,\nbulan ekstrem -11,4%", "out", num="12a")
    R = box(ax, 6.7, 2.2, 4.3, 2.6, "HASIL 2: efek iklim",
            "Odds ratio (kejadian) & rate ratio\n(jumlah) per 1 SB. Hujan lag0: OR 0,631\nHujan lag1: RR 0,823\nSuhu/RH tidak stabil -> jangan\nditafsirkan kausal", "out", num="12b")
    S = box(ax, 11.7, 2.2, 4.9, 2.6, "HASIL 3: pola laten",
            "Peta efek ruang (rawan menetap)\nProfil musim: puncak Agu-Sep\nLintasan waktu: 2015, 2019, 2023\nb ruang~0,9 ; b musim~1,0 ; b waktu~2", "out", num="12c")
    T = box(ax, 3.2, 0.2, 10.6, 1.4, "KESIMPULAN",
            "Hurdle: perbaikan terbatas, terkonsentrasi di bulan api & kalibrasi peluang kejadian.  Bukan alat prakiraan\n(iklim uji = nilai teramati). Interaksi iid (ZH2): DIC membaik tetapi CRPS tidak, biaya 7,7x.", "out", fs=10.6, num="13")

    # ---------- panah --------------------------------------------------------------
    arrow(ax, A, D); arrow(ax, B, E); arrow(ax, C, F)
    arrow(ax, D, E, "r", "l", rad=0.0)
    arrow(ax, D, G); arrow(ax, E, G, "b", "t", rad=0.25); arrow(ax, F, I, "b", "t")
    arrow(ax, E, H)
    arrow(ax, G, J); arrow(ax, H, K); arrow(ax, I, L)
    arrow(ax, J, M, "b", "t", rad=0.1); arrow(ax, K, M); arrow(ax, L, M, "b", "t", rad=-0.1)
    arrow(ax, M, N, "b", "t", rad=0.1); arrow(ax, M, O); arrow(ax, M, P, "b", "t", rad=-0.1)
    arrow(ax, N, O, "r", "l"); arrow(ax, O, P, "r", "l")
    arrow(ax, O, Q_, "b", "t", rad=0.2); arrow(ax, O, R); arrow(ax, M, S, "r", "t", rad=-0.55)
    arrow(ax, Q_, T, "b", "t", rad=0.1); arrow(ax, R, T); arrow(ax, S, T, "b", "t", rad=-0.1)
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


if __name__ == "__main__":
    matplotlib.use("Agg")
    draw_flow("alur_data_lengkap.png")
