"""
hurdle_lab.py  -  modul pendukung notebook hands-on
===================================================
Tujuan: meniru, dalam skala kecil dan murni Python (numpy/scipy), alur metode tesis
"Model Hurdle Spasio-Temporal Berbasis INLA untuk Hotspot Kebakaran di Kalimantan".

PENTING (kejujuran metode):
  * Tesis memakai paket R-INLA. INLA tidak tersedia di Python, jadi di sini kita
    menyelesaikan inti yang sama dengan cara yang lebih sederhana:
        - model Gauss laten (efek ruang BYM2, waktu AR(1), musim RW1 siklik),
        - hiperparameter DITETAPKAN (mirip "empirical Bayes" tanpa langkah optimasi),
        - posterior dihampiri dengan aproksimasi Laplace (modus + kurvatur) -
          inilah jantung INLA.
  * Jadi angka di notebook TIDAK identik dengan tesis. Yang direplikasi adalah
    LOGIKA metodenya, bukan angka-angkanya.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, expit

# ----------------------------------------------------------------------------
# 1. GRID, GRAF KETETANGGAAN (QUEEN), PRESISI ICAR / AR1 / RW1
# ----------------------------------------------------------------------------
ISLAND = [  # bentuk "pulau" 9 baris x 8 kolom (1 = daratan); sketsa kasar Kalimantan
    "00111100",
    "01111110",
    "11111111",
    "11111111",
    "11111111",
    "01111111",
    "01111110",
    "00111110",
    "00011100",
]


def make_grid():
    rows = len(ISLAND)
    cols = len(ISLAND[0])
    cells = []
    for r in range(rows):
        for c in range(cols):
            if ISLAND[r][c] == "1":
                cells.append((r, c))
    xy = np.array([(c, rows - 1 - r) for r, c in cells], float)  # x=timur, y=utara
    n = len(cells)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if max(abs(cells[i][0] - cells[j][0]), abs(cells[i][1] - cells[j][1])) == 1:
                W[i, j] = W[j, i] = 1  # queen: sisi ATAU sudut
    return xy, W


def icar_Q(W):
    return np.diag(W.sum(1)) - W


def scale_icar(Q):
    """Skala ala Riebler dkk: rata-rata geometrik ragam marjinal = 1."""
    n = Q.shape[0]
    one = np.ones((n, n)) / n
    S = np.linalg.inv(Q + one) - one
    c = np.exp(np.mean(np.log(np.diag(S))))
    return Q * c


def ar1_Q(T, rho):
    """Presisi AR(1) stasioner dengan ragam marjinal = 1."""
    Q = np.zeros((T, T))
    for t in range(T):
        Q[t, t] = (1 + rho**2) if 0 < t < T - 1 else 1.0
        if t < T - 1:
            Q[t, t + 1] = Q[t + 1, t] = -rho
    return Q / (1 - rho**2)


def rw1_cyclic_Q(m=12):
    Q = 2 * np.eye(m)
    for i in range(m):
        Q[i, (i + 1) % m] -= 1
        Q[i, (i - 1) % m] -= 1
    return Q


def scale_rw1(Q):
    return scale_icar(Q)


# ----------------------------------------------------------------------------
# 2. SIMULASI DATA (STAND-IN DATA FIRMS + NASA POWER)
# ----------------------------------------------------------------------------
def simulate(seed=2026, T=108, nlag=3):
    """Bangkitkan 'dunia' sintetis lengkap dengan kebenaran yang diketahui.
    Mengembalikan dict berisi panel sel-bulan + semua efek laten benar."""
    rng = np.random.default_rng(seed)
    xy, W = make_grid()
    n = len(xy)
    Tt = T + nlag  # bulan tambahan di depan agar lag 3 tersedia
    month = (np.arange(Tt) + 9) % 12  # 0=Jan ... ; awal data = Okt (supaya lag awal masuk akal)

    # --- efek ruang benar: medan halus (jumlah gaussian), dibakukan
    a = np.zeros(n)
    for (cx, cy, amp, w) in [(1.8, 6.5, 1.3, 1.6), (6.0, 5.0, 1.0, 1.5), (2.0, 1.5, 1.2, 1.5),
                              (4.2, 4.0, -1.8, 1.4)]:
        a += amp * np.exp(-((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) / (2 * w**2))
    a = (a - a.mean()) / a.std() * 1.0

    # --- efek musim benar (puncak Agu-Sep, siklik)
    m = np.arange(12)
    season = 1.6 * np.exp(-0.5 * (((m - 8.0 + 6) % 12 - 6) / 1.6) ** 2) - 0.85
    season += 0.4 * np.exp(-0.5 * (((m - 5.0 + 6) % 12 - 6) / 1.5) ** 2)
    season -= season.mean()

    # --- iklim: hujan = musiman + anomali tahun-ke-tahun ("El Nino") + derau sel
    year_idx = np.arange(Tt) // 12
    nyear = year_idx.max() + 1
    dry_years = {1, 4, 8}  # tahun kering "El Nino" (indeks tahun ke-)
    enso = np.array([-1.6 if y in dry_years else rng.normal(0.3, 0.45) for y in range(nyear)])
    enso_m = enso[year_idx] + 0.3 * rng.normal(size=Tt)
    base_rain = 230 + 130 * np.cos(2 * np.pi * (month - 1) / 12 + 0.2)  # basah Des-Mar
    rain_t = np.clip(base_rain + 60 * enso_m, 5, None)
    rain = np.clip(rain_t[:, None] * np.exp(0.25 * rng.normal(size=(Tt, n))) *
                   (1 + 0.15 * (xy[:, 1] - xy[:, 1].mean()) / 4)[None, :], 2, None)
    temp = 26 + 1.2 * np.sin(2 * np.pi * (month - 3) / 12)[:, None] + 0.5 * (-enso_m)[:, None] \
        + 0.3 * rng.normal(size=(Tt, n))
    rh = 85 - 4 * (-enso_m)[:, None] - 3 * np.sin(2 * np.pi * (month - 3) / 12)[:, None] \
        + 1.5 * rng.normal(size=(Tt, n))

    # --- efek waktu AR(1) benar (sisa dinamika di luar iklim)
    rho = 0.7
    u = np.zeros(Tt)
    for t in range(1, Tt):
        u[t] = rho * u[t - 1] + rng.normal(0, np.sqrt(1 - rho**2))
    u = 0.55 * u

    def lagged(x, l):
        return x[nlag - l: Tt - l]  # baris t -> nilai bulan t-l

    keep = slice(nlag, Tt)
    month_k = month[keep]
    u_k = u[keep]
    X_raw = {}
    for l in range(4):
        X_raw[f"hujan_lag{l}"] = lagged(rain, l)
    X_raw["suhu"] = temp[keep]
    X_raw["kelembapan"] = rh[keep]
    names = list(X_raw)

    # --- koefisien benar (per satu simpangan baku) - dua bagian berbeda
    beta_z = {"hujan_lag0": -0.60, "hujan_lag1": -0.30, "hujan_lag2": -0.15, "hujan_lag3": 0.0,
              "suhu": 0.0, "kelembapan": -0.10}
    beta_n = {"hujan_lag0": -0.10, "hujan_lag1": -0.28, "hujan_lag2": -0.14, "hujan_lag3": 0.0,
              "suhu": 0.0, "kelembapan": -0.30}
    # standardisasi memakai statistik "latih" (bulan 0..72) -> dikerjakan di prepare(); di sini pakai global utk simulasi
    Xs = {k: (v - v[:72].mean()) / v[:72].std() for k, v in X_raw.items()}
    b_true = dict(a=0.9, u=2.0, s=1.0)
    muz, mun = 0.15, 1.8
    area = rng.uniform(0.55, 1.0, n)  # fraksi daratan -> offset
    k_true = 0.9

    eta_z = np.zeros((T, n))
    eta_n = np.zeros((T, n))
    for t in range(T):
        lat_z = a * 1.0 + u_k[t] * 1.0 + season[month_k[t]]
        eta_z[t] = muz + lat_z + sum(beta_z[k] * Xs[k][t] for k in names)
        eta_n[t] = mun + b_true["a"] * a + b_true["u"] * u_k[t] + b_true["s"] * season[month_k[t]] \
            + sum(beta_n[k] * Xs[k][t] for k in names) + np.log(area)
    pz = expit(eta_z)
    mu = np.exp(eta_n)
    Z = rng.random((T, n)) < pz
    # y-1 ~ NB(mu, k) via gamma-Poisson
    lam = rng.gamma(k_true, mu / k_true)
    ypos = rng.poisson(lam)
    Y = np.where(Z, 1 + ypos, 0)

    return dict(xy=xy, W=W, n=n, T=T, Y=Y, month=month_k, X_raw=X_raw, names=names,
                area=area, truth=dict(a=a, u=u_k, season=season, beta_z=beta_z, beta_n=beta_n,
                                      b=b_true, k=k_true, muz=muz, mun=mun, rho=rho),
                eta_z=eta_z, eta_n=eta_n, pz=pz, mu=mu, rain=rain[keep], enso=enso)


# ----------------------------------------------------------------------------
# 3. MODEL GAUSS LATEN + APROKSIMASI LAPLACE
# ----------------------------------------------------------------------------
class LatentModel:
    """
    kind:
      'GNB' : NB regresi biasa (tanpa efek laten)           -> pembanding "tanpa laten"
      'GH'  : hurdle regresi biasa (tanpa efek laten)
      'ZN'  : NB + efek laten (ruang BYM2, AR1, musim)      -> pembanding utama
      'ZH0' : hurdle + laten, tanpa iklim
      'ZH1' : hurdle + laten + iklim                        -> model utama tesis
    """

    def __init__(self, data, train_T, kind, k_size=1.0, rho=0.7,
                 sig_a=1.0, phi=0.9, sig_u=0.6, sig_s=0.8):
        self.d, self.kind, self.train_T = data, kind, train_T
        self.hurdle = kind in ("GH", "ZH0", "ZH1")
        self.latent = kind in ("ZN", "ZH0", "ZH1")
        self.climate = kind in ("GNB", "GH", "ZN", "ZH1")
        self.k = k_size
        T, n = data["T"], data["n"]
        self.T, self.n = T, n
        self.cell = np.tile(np.arange(n), T)
        self.time = np.repeat(np.arange(T), n)
        self.mon = data["month"][self.time]
        self.y = data["Y"].reshape(-1)
        self.off = np.log(data["area"])[self.cell]
        self.is_train = self.time < train_T
        # kovariat: standardisasi pakai statistik DATA LATIH SAJA (anti-bocor)
        cols = []
        for nm in data["names"]:
            v = data["X_raw"][nm].reshape(-1)
            mu_, sd_ = v[self.is_train].mean(), v[self.is_train].std()
            cols.append((v - mu_) / sd_)
        self.Xc = np.column_stack(cols) if self.climate else np.zeros((T * n, 0))
        self.p = self.Xc.shape[1]

        # --- blok presisi komponen laten (urutan: u*(n), v(n), AR1(T), RW1(12))
        if self.latent:
            Qs = scale_icar(icar_Q(data["W"]))
            self.Qblocks = [Qs + 1e-6 * np.eye(n), np.eye(n),
                            ar1_Q(T, rho) + 1e-8 * np.eye(T),
                            scale_rw1(rw1_cyclic_Q()) + 1e-6 * np.eye(12)]
            self.sig = dict(a=sig_a, u=sig_u, s=sig_s)
            self.phi = phi
            self.nl = 2 * n + T + 12
        else:
            self.nl = 0
        self.nfix = (2 if self.hurdle else 1) * (1 + self.p)
        self.d_x = self.nfix + self.nl
        self._build_maps()

    # latent -> efek per baris -------------------------------------------------
    def _build_maps(self):
        n, T = self.n, self.T
        R = len(self.y)
        # matriks efek tetap
        Fz = np.zeros((R, 1 + self.p)); Fz[:, 0] = 1; Fz[:, 1:] = self.Xc
        self.Fz = Fz
        if self.latent:
            s_a = self.sig["a"]
            A = np.zeros((R, self.nl))
            # a_c = sig_a*(sqrt(phi) u*_c + sqrt(1-phi) v_c)
            A[np.arange(R), self.cell] = s_a * np.sqrt(self.phi)
            A[np.arange(R), n + self.cell] = s_a * np.sqrt(1 - self.phi)
            self.A_a = A.copy()
            self.A_u = np.zeros((R, self.nl)); self.A_u[np.arange(R), 2 * n + self.time] = self.sig["u"]
            self.A_s = np.zeros((R, self.nl)); self.A_s[np.arange(R), 2 * n + T + self.mon] = self.sig["s"]
            self.A_all = self.A_a + self.A_u + self.A_s
            Q = np.zeros((self.nl, self.nl))
            o = 0
            for B in self.Qblocks:
                m = B.shape[0]; Q[o:o + m, o:o + m] = B; o += m
            self.Q = Q

    def _design(self, bs):
        """Matriks desain M_z, M_n (baris x d_x) untuk skala copy b = (ba, bu, bs)."""
        R = len(self.y)
        pf = 1 + self.p
        Mz = Mn = None
        if self.hurdle:
            Mz = np.zeros((R, self.d_x)); Mz[:, :pf] = self.Fz
            Mn = np.zeros((R, self.d_x)); Mn[:, pf:2 * pf] = self.Fz
            if self.latent:
                Mz[:, self.nfix:] = self.A_all
                Mn[:, self.nfix:] = bs[0] * self.A_a + bs[1] * self.A_u + bs[2] * self.A_s
        else:
            Mn = np.zeros((R, self.d_x)); Mn[:, :pf] = self.Fz
            if self.latent:
                Mn[:, self.nfix:] = self.A_all
        return Mz, Mn

    # likelihood ---------------------------------------------------------------
    def _terms(self, x, bs):
        Mz, Mn = self._design(bs)
        tr = self.is_train
        k = self.k
        f = 0.0; g = np.zeros(self.d_x); H = np.zeros((self.d_x, self.d_x))
        if self.hurdle:
            ez = Mz @ x
            p = expit(ez)
            z = (self.y > 0).astype(float)
            ll = z * np.log(p + 1e-12) + (1 - z) * np.log(1 - p + 1e-12)
            f += -ll[tr].sum()
            gz = np.where(tr, z - p, 0.0)
            g += -(Mz.T @ gz)
            Wz = np.where(tr, p * (1 - p), 0.0)
            H += (Mz * Wz[:, None]).T @ Mz
            # bagian hitungan: hanya baris latih positif, respons y-1
            msk = tr & (self.y > 0)
            yy = np.where(msk, self.y - 1, 0).astype(float)
        else:
            msk = tr
            yy = np.where(msk, self.y, 0).astype(float)
        en = Mn @ x + self.off
        mu = np.exp(np.clip(en, -20, 15))
        ll = gammaln(yy + k) - gammaln(k) - gammaln(yy + 1) + k * np.log(k / (k + mu)) + yy * np.log(mu / (k + mu))
        f += -ll[msk].sum()
        gn = np.where(msk, (yy - mu) * k / (k + mu), 0.0)
        g += -(Mn.T @ gn)
        Wn = np.where(msk, k * mu * (k + yy) / (k + mu) ** 2, 0.0)
        H += (Mn * Wn[:, None]).T @ Mn
        return f, g, H, Mz, Mn, gn

    def fit(self):
        d_x = self.d_x
        has_b = self.hurdle and self.latent
        self.fix_prec = 1e-3  # prior Gauss presisi 0.001 untuk efek tetap (bawaan INLA)

        def neg_log_post(theta):
            x = theta[:d_x]
            bs = theta[d_x:] if has_b else np.ones(3)
            f, g, H, Mz, Mn, gn = self._terms(x, bs)
            # prior efek tetap + laten
            f += 0.5 * self.fix_prec * (x[:self.nfix] ** 2).sum()
            g[:self.nfix] += self.fix_prec * x[:self.nfix]
            if self.latent:
                xl = x[self.nfix:]
                Qxl = self.Q @ xl
                f += 0.5 * xl @ Qxl
                g[self.nfix:] += Qxl
            if has_b:
                # gradien terhadap skala copy
                gb = np.array([-(gn * (self.A_a @ x[self.nfix:])).sum(),
                               -(gn * (self.A_u @ x[self.nfix:])).sum(),
                               -(gn * (self.A_s @ x[self.nfix:])).sum()])
                # prior lemah pada b (Gauss presisi kecil) agar stabil
                f += 0.5 * 1e-2 * ((bs - 1) ** 2).sum()
                gb += 1e-2 * (bs - 1)
                g = np.concatenate([g, gb])
            return f, g

        theta0 = np.zeros(d_x + (3 if has_b else 0))
        if has_b:
            theta0[d_x:] = 1.0
        # nilai awal intersep
        pf = 1 + self.p
        pos = self.y[self.is_train & (self.y > 0)]
        if self.hurdle:
            pz = (self.y[self.is_train] > 0).mean()
            theta0[0] = np.log(pz / (1 - pz))
            theta0[pf] = np.log(max(pos.mean() - 1, 0.5))
        else:
            theta0[0] = np.log(self.y[self.is_train].mean() + 0.5)
        res = minimize(neg_log_post, theta0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=4000, maxfun=8000, ftol=1e-11, gtol=1e-6))
        self.res = res
        self.x = res.x[:d_x]
        self.bs = res.x[d_x:] if has_b else np.ones(3)
        # Laplace: kurvatur di modus -> kovarians posterior (Gauss)
        f, g, H, Mz, Mn, gn = self._terms(self.x, self.bs)
        H[:self.nfix, :self.nfix] += self.fix_prec * np.eye(self.nfix)
        if self.latent:
            H[self.nfix:, self.nfix:] += self.Q
        self.Sigma = np.linalg.inv(H + 1e-9 * np.eye(d_x))
        self.Mz, self.Mn = Mz, Mn
        return self

    # efek yang diestimasi -----------------------------------------------------
    def effects(self):
        n, T = self.n, self.T
        xl = self.x[self.nfix:]
        out = {}
        if self.latent:
            out["a"] = self.sig["a"] * (np.sqrt(self.phi) * xl[:n] + np.sqrt(1 - self.phi) * xl[n:2 * n])
            out["u"] = self.sig["u"] * xl[2 * n:2 * n + T]
            out["s"] = self.sig["s"] * xl[2 * n + T:]
        pf = 1 + self.p
        if self.hurdle:
            out["fix_z"], out["fix_n"] = self.x[:pf], self.x[pf:2 * pf]
        else:
            out["fix_n"] = self.x[:pf]
        out["b"] = self.bs
        return out

    # prediksi probabilistik ----------------------------------------------------
    def predict_samples(self, S=300, seed=1):
        rng = np.random.default_rng(seed)
        te = ~self.is_train
        idx = np.where(te)[0]
        res = {}
        k = self.k
        def draw_eta(M):
            m = M[idx] @ self.x
            v = np.einsum("ij,jk,ik->i", M[idx], self.Sigma, M[idx])
            return m[:, None] + np.sqrt(np.maximum(v, 1e-12))[:, None] * rng.normal(size=(len(idx), S))
        en = draw_eta(self.Mn) + self.off[idx][:, None]
        mu = np.exp(np.clip(en, -20, 15))
        lam = rng.gamma(k, mu / k)
        cnt = rng.poisson(lam)
        if self.hurdle:
            pi = expit(draw_eta(self.Mz))
            z = rng.random(pi.shape) < pi
            ysamp = np.where(z, 1 + cnt, 0)
            res["pi"] = pi
        else:
            ysamp = cnt
            res["pi"] = None
        res["y"], res["idx"] = ysamp, idx
        return res


# ----------------------------------------------------------------------------
# 4. METRIK EVALUASI
# ----------------------------------------------------------------------------
def crps_samples(samp, obs):
    """CRPS dari sampel prediktif (S kolom). Lebih kecil = lebih baik."""
    S = samp.shape[1]
    xs = np.sort(samp, axis=1)
    term1 = np.abs(samp - obs[:, None]).mean(1)
    i = np.arange(1, S + 1)
    term2 = (xs * (2 * i - S - 1)).sum(1) / S**2
    return term1 - term2


def pit_randomized(samp, obs, rng):
    below = (samp < obs[:, None]).mean(1)
    atorbelow = (samp <= obs[:, None]).mean(1)
    return below + rng.random(len(obs)) * (atorbelow - below)


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot, e = len(p), 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.sum() / tot * abs(p[m].mean() - y[m].mean())
    return e


def block_bootstrap_ci(d, B=1000, block=3, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    nb = int(np.ceil(n / block))
    means = []
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, nb)
        samp = np.concatenate([d[s:s + block] for s in starts])[:n]
        means.append(samp.mean())
    return np.percentile(means, [2.5, 97.5])
