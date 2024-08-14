def adjusted_r_squared(r_squared, n, k):
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))
    return adjusted_r_squared

r_squared_value = 0.75 
n_obs = 100 
n_pred = 3 
result = adjusted_r_squared(r_squared_value, n_obs, n_pred) 

print("Adjusted R-squared:", result)