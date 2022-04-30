import pandas as pd
import numpy as np
# import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




if __name__ == '__main__':

    data_list = sorted(Path(str(Path(os.getcwd()).parent.absolute()) + '/ProcessedHistograms/Kenya').glob('*.npy'))
    for i in data_list:
        data_array = np.load(i.absolute())
        data = pd.DataFrame(data_array)
        print(data.shape)
        plt.imshow(data.to_numpy(), cmap='hot')
        # plt.colorbar()
        plt.show()