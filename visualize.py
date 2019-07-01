import os
import glob
import random
import pandas as pd
import matplotlib.pyplot as plt


v = '1'
files = glob.glob('predict_mc/results{}/density*.csv'.format(v))

num_files = 10

for i in range(num_files):
    csv_file = random.choice(files)
    df = pd.read_csv(csv_file, index_col=0)
    df.drop
    print(df)
    df.plot()
    plt.title('File: {}'.format(csv_file))
    plt.show()
