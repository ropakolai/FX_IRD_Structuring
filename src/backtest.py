import numpy as np

def discount_factor(rate, t):
    """Facteur d'actualisation simple"""
    return np.exp(-rate * t)

def linear_curve(times, start_rate=0.02, slope=0.001):
    """Courbe de taux linéaire simulée"""
    return [start_rate + slope*t for t in times]