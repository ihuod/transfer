import numpy as np
from scipy.stats import spearmanr

mse_model = np.mean((y_test - y_pred)**2)
mse_mean  = np.mean((y_test - y_test.mean())**2)
spearman  = spearmanr(y_test, y_pred).correlation
print({'r2': 1 - mse_model/mse_mean, 'mse_model': mse_model, 'mse_mean': mse_mean, 'spearman': spearman})
