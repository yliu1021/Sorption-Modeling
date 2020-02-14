# model_name = 'vae_conditional'

train_dir = '../generative_model_3/step_0/'
test_dir = '../generative_model_3/step_0'

# Target density for visualization
import numpy as np
target_density = np.genfromtxt('../generative_model_3/step_0/results/density_0000.csv', delimiter=',')[:40].reshape(1, 40)


def run_dft():
    if os.system('../cpp/fast_dft ./{}/'.format(model_name)):
        print('Failed to execute dft, no results found')