import numpy as np

def simulate_rates(initial_rate=0.02, vol=0.01, T=5, steps=50, n_scenarios=1000):
    """Simule des courbes de taux via un processus brownien géométrique"""
    dt = T/steps
    rates = np.zeros((n_scenarios, steps+1))
    rates[:,0] = initial_rate
    for t in range(1, steps+1):
        dz = np.random.normal(0, np.sqrt(dt), n_scenarios)
        rates[:,t] = rates[:,t-1] + vol*rates[:,t-1]*dz
    return rates

def simulate_fx(S0=1.10, vol=0.12, T=1, steps=50, n_scenarios=1000):
    """Simule un FX spot via un GBM simple"""
    dt = T/steps
    S = np.zeros((n_scenarios, steps+1))
    S[:,0] = S0
    for t in range(1, steps+1):
        dz = np.random.normal(0, np.sqrt(dt), n_scenarios)
        S[:,t] = S[:,t-1] * np.exp(-0.5*vol**2*dt + vol*dz)
    return S