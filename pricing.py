import numpy as np
from scipy.stats import norm
from src.utils import discount_factor

# ---------------- IRS ----------------
def npv_fixed_leg(notional, fixed_rate, times, discount_curve):
    pv = 0
    for t in times:
        df = discount_curve(t)
        pv += fixed_rate * notional * df
    return pv

def npv_floating_leg(notional, forward_rates, times, discount_curve):
    pv = 0
    for i, t in enumerate(times):
        df = discount_curve(t)
        pv += forward_rates[i] * notional * df
    return pv

def irs_npv(notional, fixed_rate, forward_rates, times, discount_curve):
    pv_fixed = npv_fixed_leg(notional, fixed_rate, times, discount_curve)
    pv_float = npv_floating_leg(notional, forward_rates, times, discount_curve)
    return pv_float - pv_fixed

def dv01(irs_npv_func, rate_shift=0.0001, *args, **kwargs):
    pv0 = irs_npv_func(*args, **kwargs)
    # shift curve
    shifted_curve = lambda t: kwargs['discount_curve'](t) + rate_shift
    pv1 = irs_npv_func(*args, **kwargs, discount_curve=shifted_curve)
    return pv1 - pv0

# ---------------- FX Option ----------------
def fx_option_price(S, K, T, r_dom, r_for, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r_dom - r_for + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S * np.exp(-r_for*T) * norm.cdf(d1) - K*np.exp(-r_dom*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r_dom*T)*norm.cdf(-d2) - S*np.exp(-r_for*T)*norm.cdf(-d1)
    return price

# ---------------- Cross-Currency Swap ----------------
def ccs_npv(notional_dom, notional_for, fixed_rate_dom, fixed_rate_for, times, 
            discount_curve_dom, discount_curve_for, fx_rate):
    pv_dom = npv_fixed_leg(notional_dom, fixed_rate_dom, times, discount_curve_dom)
    pv_for = npv_fixed_leg(notional_for, fixed_rate_for, times, discount_curve_for)
    pv_for_dom = pv_for * fx_rate
    return pv_dom - pv_for_dom