model_name = 'vae_conditional'

# Target density for visualization
import numpy as np
target_density = np.genfromtxt('../generative_model_3/step_0/results/density_0000.csv', delimiter=',')[:40].reshape(1, 40)