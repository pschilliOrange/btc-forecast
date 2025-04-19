import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize

# -------------------------------------------------
# 1. Market data
S0 = 84.5  # today’s BTC spot
barriers = np.array([100, 95, 90, 70], dtype=float)
market_probs = np.array([0.03, 0.12, 0.37, 0.07])  # one‑touch prices ≈ probabilities

# -------------------------------------------------
# 2. Kou jump‑diffusion simulator
def simulate_paths(params, n_paths=20000, steps=12, seed=None):
    seed = None
    """Simulate log‑price paths on a daily grid for the Kou model."""
    mu, sigma, lam, p_up, eta1, eta2 = params
    rng = np.random.default_rng(seed)
    dt = 1 / 365  # one trading day
    # drift correction so that E[e^{Y}] − 1 is compensated
    exp_jump = p_up * eta1 / (eta1 - 1) + (1.0 - p_up) * eta2 / (eta2 + 1)
    drift_corr = -lam * (exp_jump - 1)

    # Brownian increments
    Z = rng.standard_normal((n_paths, steps)) * sigma * np.sqrt(dt)

    # Jump increments (≤ 1‑jump approximation per dt – fine for daily grid)
    jump_flag = rng.random((n_paths, steps)) < lam * dt
    up_flag = rng.random((n_paths, steps)) < p_up
    Y = np.zeros_like(Z)
    # upward jumps
    mask_up = jump_flag & up_flag
    Y[mask_up] = rng.exponential(scale=1 / eta1, size=mask_up.sum())
    # downward jumps
    mask_dn = jump_flag & (~up_flag)
    Y[mask_dn] = -rng.exponential(scale=1 / eta2, size=mask_dn.sum())

    dlogS = (mu + drift_corr) * dt + Z + Y
    log_paths = np.cumsum(dlogS, axis=1)
    S = S0 * np.exp(log_paths)
    S = np.concatenate([np.full((n_paths, 1), S0), S], axis=1)  # prepend day‑0
    return S

def first_passage(params, barriers, steps=12, n_paths=20000, seed=120):
    """Monte‑Carlo estimate of first‑passage probabilities for given barriers."""
    S = simulate_paths(params, n_paths=n_paths, steps=steps, seed=seed)
    probs = []
    for B in barriers:
        if B > S0:                  # upper barrier
            hit = (S[:, 1:] >= B).any(axis=1)
        else:                       # lower barrier
            hit = (S[:, 1:] <= B).any(axis=1)
        probs.append(hit.mean())
    return np.array(probs)

# -------------------------------------------------
# 3. Calibration by minimising squared error
def objective(x):
    mu, sigma, lam, p_up, eta1, eta2 = x
    # quick parameter sanity
    if sigma <= 0 or lam < 0 or p_up <= 0 or p_up >= 1 or eta1 <= 1 or eta2 <= 0:
        return 1e6
    model = first_passage(x, barriers, n_paths=6000, seed=321)
    return np.sum((model - market_probs) ** 2)

bounds = [(-0.2, 0.2),       # mu
          (0.05, 1.5),       # sigma
          (0.0, 5.0),        # lambda
          (0.01, 0.99),      # p_up
          (1.1, 20.0),       # eta1  (>1 ensures E[e^Y] finite)
          (1.0, 20.0)]       # eta2
result = optimize.differential_evolution(
            objective, bounds, maxiter=30, popsize=12, polish=True, seed=7)
params_hat = result.x
mu, sigma, lam, p_up, eta1, eta2 = params_hat

# refined one‑touch probabilities
model_probs = first_passage(params_hat, barriers, n_paths=30000, seed=42)

# -------------------------------------------------
# 4. Forward distribution snapshots
snap_days = [0, 3, 6, 9, 12]
paths = simulate_paths(params_hat, n_paths=40000, steps=12, seed=99)
price_grid = np.linspace(50, 140, 500)

pdfs = {}
for d in snap_days:
    if d == 0:
        # day‑0 is a point mass; approximate with a very tight normal for plotting
        pdfs[d] = stats.norm.pdf(price_grid, loc=S0, scale=0.01)
    else:
        kde = stats.gaussian_kde(paths[:, d])
        pdfs[d] = kde(price_grid)

# -------------------------------------------------
# 5. Plot PDFs
plt.figure(figsize=(10, 6))
for d in snap_days:
    plt.plot(price_grid, pdfs[d], label=f"Day {d}")
plt.title("Kou‑model marginal PDFs calibrated to one‑touch quotes")
plt.xlabel("BTC price")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 6. Textual calibration report
print("\nFitted Kou‑model parameters:")
for name, val in zip(["mu", "sigma", "lambda", "p_up", "eta1", "eta2"], params_hat):
    print(f"  {name:<6}= {val: .5f}")

comparison = pd.DataFrame({
    "Barrier": barriers,
    "Market Prob": market_probs,
    "Model Prob": model_probs
})
print("\nOne‑touch probabilities: market vs. model")
print(comparison.to_string(index=False))


from scipy.stats import poisson

# plug in your calibrated params here
mu, sigma, lam, p, eta1, eta2 = params_hat
S0 = 85.0

# precompute jump moments
E_Y   =  p/eta1 - (1-p)/eta2
Var_Y = p*(2/eta1**2) + (1-p)*(2/eta2**2) - E_Y**2

for d in [0, 3, 6, 9, 12]:
    lam_eff = lam * d/365
    pois    = poisson(lam_eff)

    # pick a cutoff so that cumulative mass ≥99.9%
    k_max = int(pois.ppf(0.999))
    ks    = list(range(k_max+1))
    wts   = pois.pmf(ks)

    print(f"\nDay {d}  →  Poisson(λT) cutoff at k={k_max}, components={len(ks)}")
    print("  k\tweight\t\tmean_k\t\tvar_k")
    for k, w in zip(ks, wts):
        m_k = np.log(S0) \
            + (mu - 0.5*sigma**2 - lam*(p*eta1/(eta1-1)+(1-p)*eta2/(eta2+1)-1))*(d/365) \
            + k*E_Y
        v_k = sigma**2*(d/365) + k*Var_Y
        print(f"  {k}\t{w:.4f}\t\t{m_k:.4f}\t\t{v_k:.4f}")


