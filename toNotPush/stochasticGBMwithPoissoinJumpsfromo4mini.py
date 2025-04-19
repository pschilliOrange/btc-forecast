import numpy as np
import scipy.optimize as opt
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# --- USER INPUTS ----------------------------------------------------------
S0            = 85.0
barriers      = np.array([100, 95, 90, 70])
market_prices = np.array([0.04, 0.15, 0.42, 0.08])
T_days        = 12
r             = 0.0       # risk-free rate (assumed zero)
N_calib       = 10   # paths for calibration
N_pdf         = 200    # paths for final pdf estimation
dt_days       = 0.5       # time step in days for simulation
# --------------------------------------------------------------------------

T_years = T_days / 365.0
dt_years = dt_days / 365.0
n_steps = int(np.ceil(T_days / dt_days))


def simulate_jump_diffusion(params, n_paths, record_times_days):
    """
    Simulate GBM + compound Poisson (double-exponential jumps).
    - ensures day 0 is always recorded
    - returns:
        S_paths: dict t_day -> array of S_t
        hit:    (n_paths, len(barriers)) boolean: barrier hit by T_days
    """
    mu, sigma, lam, p_up, eta1, eta2 = params

    # Always include t=0
    record_days = sorted(set(record_times_days) | {0})
    # Map from step index -> day
    step_to_day = {int(day/dt_days): day for day in record_days}

    # Prepare storage
    S_paths = {day: np.zeros(n_paths) for day in record_days}
    hit = np.zeros((n_paths, len(barriers)), dtype=bool)

    for i in range(n_paths):
        S = S0
        # record at t=0
        S_paths[0][i] = S0

        for step in range(1, n_steps+1):
            # 1) diffusion
            z = np.random.randn()
            S *= np.exp((mu - 0.5*sigma**2)*dt_years + sigma*np.sqrt(dt_years)*z)
            # 2) jump?
            if np.random.rand() < lam * dt_years:
                if np.random.rand() < p_up:
                    J = np.random.exponential(1/eta1)
                else:
                    J = -np.random.exponential(1/eta2)
                S *= np.exp(J)
            # 3) check barrier hits
            for j, B in enumerate(barriers):
                if not hit[i,j]:
                    if B > S0 and S >= B:
                        hit[i,j] = True
                    elif B < S0 and S <= B:
                        hit[i,j] = True
            # 4) record if this step corresponds to a record day
            if step in step_to_day:
                day = step_to_day[step]
                S_paths[day][i] = S

    return S_paths, hit


def barrier_probabilities(params):
    """Estimate one‑touch probabilities at expiration T_days."""
    _, hit = simulate_jump_diffusion(params, N_calib, [T_days])
    return hit.mean(axis=0)


def loss_fn(params):
    # invalid parameter penalty
    print('params', params)
    mu, sigma, lam, p_up, eta1, eta2 = params
    if sigma <= 0 or lam <= 0 or eta1 <= 0 or eta2 <= 0 or not (0 < p_up < 1):
        return 1e3
    model_probs = barrier_probabilities(params)
    return np.sum((model_probs - market_prices)**2)


# --- calibrate ------------------------------------------------------------
x0 = np.array([0.00, 0.5, 1.0, 0.5, 10.0, 10.0])
bounds = [(-1,1), (1e-3,2), (1e-3,5), (1e-3,1-1e-3), (1,50), (1,50)]

print("Calibrating model parameters to one-touch prices...")
res = opt.minimize(loss_fn, x0, bounds=bounds, method='L-BFGS-B')
mu, sigma, lam, p_up, eta1, eta2 = res.x
print("Calibration complete.\n")

# --- diagnostics ----------------------------------------------------------
print("Calibrated parameters:")
print(f"  mu     = {mu:.4f}")
print(f"  sigma  = {sigma:.4f}")
print(f"  lambda = {lam:.4f} jumps/yr")
print(f"  p_up   = {p_up:.4f}")
print(f"  eta1   = {eta1:.4f}, eta2 = {eta2:.4f}")

model_probs = barrier_probabilities(res.x)
print("\nOne‐touch probs @ T=12 days (model vs market):")
for B, mk, md in zip(barriers, market_prices, model_probs):
    print(f"  Barrier {B:>5.0f}: market {mk:.3f}, model {md:.3f}")

# --- PDF estimation & plotting ------------------------------------------
print("\nSimulating to estimate PDFs at days 0,3,6,9,12...")
times = [0, 3, 6, 9, 12]
S_paths, _ = simulate_jump_diffusion(res.x, N_pdf, times)

plt.figure(figsize=(10,6))
x_grid = np.linspace(0, 200, 500)
for t in times:
    data = S_paths[t]
    kde = gaussian_kde(data, bw_method='scott')
    plt.plot(x_grid, kde(x_grid), label=f"t = {t} days")

plt.title("Forecasted BTC Price PDF under Jump–Diffusion")
plt.xlabel("BTC Price")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
