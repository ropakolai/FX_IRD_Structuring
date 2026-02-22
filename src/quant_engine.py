import numpy as np
from scipy.stats import norm


# ================================
# HULL WHITE 1F
# ================================

class HullWhite1F:

    def __init__(self, a=0.1, sigma=0.01):
        self.a = a
        self.sigma = sigma

    def calibrate_theta(self, curve, dt):

        n = len(curve)
        theta = np.zeros(n)

        for i in range(1,n):
            theta[i] = (
                (curve[i]-curve[i-1])/dt
                + self.a*curve[i-1]
            )

        return theta

    def simulate(self, r0, theta, dt, n_steps, n_scenarios):

        rates = np.zeros((n_scenarios,n_steps))
        rates[:,0] = r0

        for t in range(1,n_steps):

            z = np.random.normal(size=n_scenarios)

            dr = (
                (theta[t]-self.a*rates[:,t-1])*dt
                + self.sigma*np.sqrt(dt)*z
            )

            rates[:,t] = rates[:,t-1] + dr

        return rates


# ================================
# FX GBM MODEL
# ================================

class FXModel:

    def __init__(self, vol):
        self.vol = vol

    def simulate(self, S0, r_dom, r_for, corr_z, dt):

        n_scenarios, n_steps = r_dom.shape

        fx = np.zeros((n_scenarios,n_steps))
        fx[:,0] = S0

        for t in range(1,n_steps):

            drift = (r_dom[:,t-1]-r_for[:,t-1]-0.5*self.vol**2)*dt
            diffusion = self.vol*np.sqrt(dt)*corr_z[:,t-1]

            fx[:,t] = fx[:,t-1]*np.exp(drift + diffusion)

        return fx


# ================================
# PRICING
# ================================

def irs_price_mc(rates, notional, fixed_rate, maturity_idx, discount_curve):

    pnls = []

    for path in rates:

        df = np.exp(-np.cumsum(path)*0.02)

        float_leg = np.mean(path[maturity_idx])
        fixed_leg = fixed_rate*len(maturity_idx)

        price = notional*(float_leg-fixed_leg)*np.mean(df)

        pnls.append(price)

    return np.array(pnls)


def fx_option_bs(S,K,T,rd,rf,sigma,call=True):

    d1 = (np.log(S/K)+(rd-rf+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if call:
        return np.exp(-rd*T)*(S*np.exp((rd-rf)*T)*norm.cdf(d1)-K*norm.cdf(d2))
    else:
        return np.exp(-rd*T)*(K*norm.cdf(-d2)-S*np.exp((rd-rf)*T)*norm.cdf(-d1))


def var_es(x, alpha=0.05):

    var = np.percentile(x,alpha*100)
    es = np.mean(x[x<=var])

    return var, es
