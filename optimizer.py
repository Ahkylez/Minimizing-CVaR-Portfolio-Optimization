import numpy as np
import yfinance as yf
import cvxpy as cp

def get_returns(tickers, period='5y', interval='1d'):
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=False)
    returns = data['Close'].pct_change().dropna()
    returns = returns.reindex(columns=tickers)
    return returns


tickers = [
    'AAPL',
    'JNJ',
    'XOM',
    'WMT',
    'JPM',
    'BA',
    'DD',
    'AMT',
    'GOOGL',
    'DUK',
    'AMZN'
    ] 

n = len(tickers)

returns = get_returns(tickers)
m = returns.mean()
V = returns.cov()

# Generate samples from normal distribution, q samples N(m, V)
q = 10000
y = np.random.multivariate_normal(m, V, q) 

beta = 0.99
x = cp.Variable(n, nonneg=True) # Portfolio weights 
alpha = cp.Variable()           # VaR approx
u = cp.Variable(q, nonneg=True) # Auxiliary variables 

# Constraints
constraints = [
    cp.sum(x) == 1,
    u >= -y @ x - alpha
]

# CVaR Minimization
# CVaR =  alpha + 1/(q(1-beta)) * sum(u_k)
objective = cp.Minimize(alpha + (1/(q*(1-beta))) * cp.sum(u))


prob = cp.Problem(objective, constraints)
try:
    prob.solve(solver=cp.CLARABEL)
    
    print("Optimization Status:", prob.status)
    
    if prob.status in ["optimal", "optimal_inaccurate"]:
        
        print("Optimal alpha (VaR at 99%):", alpha.value)
        cvar = prob.value
        print("Optimal CVaR (Expected Loss beyond VaR):", cvar)
        print(f'With probability {beta}, loss will be less than {alpha.value*100:.2f}%')

        # Map weights to tickers and pretty-print
        weights = np.asarray(x.value).flatten()
        pairs = list(zip(tickers, weights))
        pairs_filtered = [(t, w) for t, w in pairs if w > 1e-6] 
        pairs_sorted = sorted(pairs_filtered, key=lambda t: t[1], reverse=True)

        print("\nOptimal portfolio weights (non-zero):")
        for tkr, w in pairs_sorted:
            print(f"  {tkr:<6}  {w:>8.4%}")

        print(f"\nSum of weights: {weights.sum():.6f}")
    else:
        print("The optimization problem could not be solved to optimality.")

except Exception as e:
    print(f"\nAn error occurred during problem solving: {e}")