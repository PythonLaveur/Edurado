import numpy as np

def ab(m, v, scale = 20):
    m /= scale
    v = v**2 /scale**2
    a = (m**2 - m**3)/v - m
    return a, a/m - a
def generate_vector(w, n, admissible_values, mult):
    admissible_values = (admissible_values*mult).astype(int)
    w *= mult
    w2 = w
    tolerance = 2
    k = 0
    for k in range(100):
        vector = []
        w = w2
        while len(vector) < n:
            valid_values = [v for v in admissible_values if w - v >= (n - len(vector) - 1)*min(admissible_values)]
            if not valid_values:
                vector = [0] * n
            else:
                pick = np.random.choice(valid_values)
                vector.append(pick)
                w -= pick
    return np.array(vector)/mult

def simulate_weights(sum_weight, sw_std, n_weight, n_std, admissible_values, mult = 20):
    w = round(np.clip(np.random.gamma(sum_weight**2/sw_std**2, sw_std**2/sum_weight), 2, 10)*mult)/mult
    n = round(np.clip(np.random.gamma(sum_weight**2/sw_std**2, sw_std**2/sum_weight), max(1, w/3) ,10))
    if n ==1:
        i = np.argmin(np.abs(admissible_values - w))
        return admissible_values[i]* np.ones(1)
    return generate_vector(w, n, admissible_values, mult)

def vanilla_call_payoff(x, k, r, t):
    return np.maximum(x - k, 0)
def price_vanilla_call(target, indiv_vol, macro_vol, k, r, t, scale = 20, n_samples = 1e4):
    k = np.asarray(k).reshape(-1, 1)
    alpha, beta = ab(target, indiv_vol, scale)
    x = np.random.beta(alpha, beta, n_samples)*scale
    m = np.random.normal(0, macro_vol, n_samples)
    x = np.clip(x + m, 0, scale)
    return max(0.01, np.mean(vanilla_call_payoff(x, k, r, t), axis = 1))* np.exp(-r*t)
def squared_call_payoff(x ,k, r, t):
    return np.maximum(x - k, 0)**2
def price_squared_call(target, indiv_vol, macro_vol, k, r, t, scale = 20, n_samples = 1e4):
    k = np.asarray(k).reshape(-1, 1)
    alpha, beta = ab(target, indiv_vol, scale)
    x = np.random.beta(alpha, beta, n_samples)*scale
    m = np.random.normal(0, macro_vol, n_samples)
    x = np.clip(x + m, 0, scale)
    return max(0.01, np.mean(squared_call_payoff(x, k, r, t), axis = 1))* np.exp(-r*t)
def mean_spread_payoff(x, z, k, r, t):
    return np.maximum(x - z - k, 0)
def price_mean_spread(target, indiv_vol, mean, mean_vol, k, r, t, scale = 20, n_samples = 1e4):
    k = np.asarray(k).reshape(-1, 1)
    alpha, beta = ab(target, indiv_vol, scale)
    alpha2, beta2 = ab(mean, mean_vol, scale)
    x = np.random.beta(alpha, beta, n_samples)*scale
    z = np.random.beta(alpha2, beta2, n_samples)*scale
    return max(0.01, np.mean(mean_spread_payoff(x,z, k, r, t), axis = 1))* np.exp(-r*t)
def high_spread_payoff(x, xh, k, r, t):
    return np.maximum(x - xh - k, 0)
def price_high_spread(target, indiv_vol, high, high_vol, k, r, t, scale = 20, n_samples = 1e4):
    k = np.asarray(k).reshape(-1, 1)
    alpha, beta = ab(target, indiv_vol, scale)
    alpha2, beta2 = ab(high, high_vol, scale)
    x = np.random.beta(alpha, beta, n_samples)*scale
    xh = np.random.beta(alpha2, beta2, n_samples)*scale
    xh = np.maximum(xh, x)
    return max(0.01, np.mean(high_spread_payoff(x,xh, k, r, t), axis = 1))* np.exp(-r*t)
def max_in_payoff(x, weights, k, r, t):
    coefs_matrix = np.tile(weights, (k.shape[0], 1))
    in_index = x > k
    if in_index.ndim == 1:
        in_index = in_index[np.newaxis, :]
    coefs_matrix[~in_index] = 0
    return np.minimum(np.sum(coefs_matrix, axis = 1), 1)
def price_max_in(target, unit_vol, macro_vol, effort_vol, sum_weight, sw_std, n_weight, n_std, k ,r, t, scale = 20, n_samples = 1e4):
    k = np.asarray(k).reshape(-1, 1)
    alpha, beta = ab(target, unit_vol, scale)
    payoffs = np.zeros((n_samples, k.shape[0]))
    for i in range(n_samples):
        w = simulate_weights(sum_weight, sw_std, n_weight, n_std)
        x = np.random.beta(alpha, beta, w.shape[0])*scale
        m = np.random.normal(0, macro_vol, w.shape[0])
        m2 = np.random.normal(0, effort_vol, w.shape[0])
        x = np.clip(x + m + m2, 0, scale)
        payoffs[i, :] = max_in_payoff(x, w, k, r,t)
    return max(0.01, np.mean(payoffs, axis = 0)) * np.exp(-r*t)



