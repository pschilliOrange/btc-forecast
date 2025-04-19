import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt

# Constants
S0 = 85.0  # Current asset price
T = 12 / 365  # Expiration in years (12 days)
r = 0.05  # Risk-free rate (assumed)
barriers = [100, 95, 90, 70]  # Option barriers
market_prices = [0.04, 0.15, 0.42, 0.08]  # Market prices of one-touch options
payoff = 1.0  # Payoff if barrier is hit
n_simulations = 1000  # Number of Monte Carlo paths
n_steps = 1000  # Time steps for path simulation

# Simulate regime-switching GBM paths
def simulate_regime_switching_paths(S0, T, mu1, sigma1, mu2, sigma2, lambda12, lambda21, p1_0, n_simulations, n_steps):
    """
    Simulate asset price paths under regime-switching GBM.
    Returns array of final prices and paths for option pricing.
    """
    dt = T / n_steps
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    regimes = np.zeros((n_simulations, n_steps + 1), dtype=int)  # 1 or 2
    regimes[:, 0] = np.random.choice([1, 2], size=n_simulations, p=[p1_0, 1 - p1_0])
    
    for t in range(n_steps):
        # Transition probabilities
        prob_1_to_2 = lambda12 * dt
        prob_2_to_1 = lambda21 * dt
        transitions = np.random.uniform(0, 1, n_simulations)
        regimes[:, t + 1] = regimes[:, t]
        regimes[:, t + 1][(regimes[:, t] == 1) & (transitions < prob_1_to_2)] = 2
        regimes[:, t + 1][(regimes[:, t] == 2) & (transitions < prob_2_to_1)] = 1
        
        # Apply GBM in current regime
        z = np.random.normal(0, 1, n_simulations)
        for i in range(n_simulations):
            mu = mu1 if regimes[i, t] == 1 else mu2
            sigma = sigma1 if regimes[i, t] == 1 else sigma2
            paths[i, t + 1] = paths[i, t] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[i]
            )
    
    return paths

# Price one-touch options via Monte Carlo
def one_touch_price_mc(S0, B, T, mu1, sigma1, mu2, sigma2, lambda12, lambda21, p1_0, payoff=1.0):
    """
    Price a one-touch option using Monte Carlo under regime-switching GBM.
    """
    paths = simulate_regime_switching_paths(
        S0, T, mu1, sigma1, mu2, sigma2, lambda12, lambda21, p1_0, n_simulations, n_steps
    )
    if B > S0:  # Up-and-in
        hits = np.any(paths >= B, axis=1)
    else:  # Down-and-in
        hits = np.any(paths <= B, axis=1)
    price = payoff * np.mean(hits) * np.exp(-r * T)  # Discounted payoff
    return price

# Objective function for calibration
def objective(params, S0, barriers, T, market_prices):
    """
    Compute squared error between market and model prices.
    params = [mu1, sigma1, mu2, sigma2, lambda12, lambda21, p1_0]
    """
    mu1, sigma1, mu2, sigma2, lambda12, lambda21, p1_0 = params
    model_prices = [
        one_touch_price_mc(S0, B, T, mu1, sigma1, mu2, sigma2, lambda12, lambda21, p1_0, payoff)
        for B in barriers
    ]
    return np.sum((np.array(model_prices) - np.array(market_prices))**2)

# Calibrate regime-switching GBM
def calibrate_regime_switching_gbm(S0, barriers, T, market_prices):
    """
    Calibrate parameters using numerical optimization.
    """
    # Initial guess: bullish regime (mu1=0.2, sigma1=0.3), bearish (mu2=-0.2, sigma2=0.3)
    initial_guess = [0.2, 0.3, -0.2, 0.3, 1.0, 1.0, 0.5]
    # Bounds: mu in [-1, 1], sigma in [0.01, 1], lambda in [0, 10], p1_0 in [0, 1]
    bounds = [(-1, 1), (0.01, 1), (-1, 1), (0.01, 1), (0, 10), (0, 10), (0, 1)]
    
    result = minimize(
        objective,
        initial_guess,
        args=(S0, barriers, T, market_prices),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100}
    )
    
    if result.success:
        mu1, sigma1, mu2, sigma2, lambda12, lambda21, p1_0 = result.x
        return mu1, sigma1, mu2, sigma2, lambda12, lambda21, p1_0
    else:
        raise ValueError("Optimization failed: " + result.message)

# Compute PDF at specified time
def get_price_pdf(day, S0, T, barriers, market_prices):
    """
    Calibrate model and compute PDF for asset price on the specified day.
    """
    t = day / 365
    params = calibrate_regime_switching_gbm(S0, barriers, T, market_prices)
    mu1, sigma1, mu2, sigma2, lambda12, lambda21, p1_0 = params
    print(f"Calibrated parameters:")
    print(f"Bullish regime: mu1 = {mu1:.4f}, sigma1 = {sigma1:.4f}")
    print(f"Bearish regime: mu2 = {mu2:.4f}, sigma2 = {sigma2:.4f}")
    print(f"Transition rates: lambda12 = {lambda12:.4f}, lambda21 = {lambda21:.4f}")
    print(f"Initial prob: p1_0 = {p1_0:.4f}")
    
    # Simulate paths to time t
    paths = simulate_regime_switching_paths(
        S0, t, mu1, sigma1, mu2, sigma2, lambda12, lambda21, p1_0, n_simulations, max(100, int(n_steps * t / T))
    )
    final_prices = paths[:, -1]
    
    # Estimate PDF using kernel density estimation
    kde = gaussian_kde(final_prices)
    price_range = np.linspace(50, 120, 500)
    pdf_values = kde(price_range)
    
    # Plot the PDF
    plt.figure(figsize=(10, 6))
    plt.plot(price_range, pdf_values, label=f'PDF at day {day}')
    plt.title(f'Asset Price PDF at Day {day} (t = {t:.4f} years)')
    plt.xlabel('Asset Price')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return price_range, pdf_values

# User input and execution
if __name__ == "__main__":
    while True:
        try:
            day = int(input("Enter a day (1 to 12) to compute the PDF: "))
            if 1 <= day <= 12:
                break
            else:
                print("Please enter a day between 1 and 12.")
        except ValueError:
            print("Please enter a valid integer.")
    
    prices, pdf = get_price_pdf(day, S0, T, barriers, market_prices)
    
    # Print PDF values at select points
    print(f"\nPDF values at day {day}:")
    for price in [70, 85, 90, 95, 100]:
        idx = np.argmin(np.abs(prices - price))
        print(f"Price = {price:.2f}, PDF = {pdf[idx]:.6f}")