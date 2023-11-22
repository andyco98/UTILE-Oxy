import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def moving_average_and_std(series, window_size):
    series = series.to_numpy()  # Convert series to numpy array
    ret = np.cumsum(series, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    moving_avg = ret[window_size - 1:] / window_size
    moving_std = [series[i:i+window_size].std() for i in range(len(series) - window_size + 1)]
    return moving_avg, moving_std


def moving_average_individual_analysis(csv_directory):

    fig, ax = plt.subplots(4,2, figsize=(12,12), sharex=True)


    df = pd.read_csv(csv_directory)

    window_size = 100  # Calculate moving average and std every 100 datapoints
    x = np.arange(window_size - 1, len(df))

    rows = 0
    cols = 0


    for column in df.columns:
        moving_avg, moving_std = moving_average_and_std(df[column], window_size)

        mean_value = np.mean(moving_avg)
        ax[rows,cols].plot(x, moving_avg, label=f'Mean value: {mean_value:.2f}')
        ax[rows,cols].fill_between(x, moving_avg - moving_std, moving_avg + moving_std, alpha=0.2)
        #ax[rows,cols].set_xlabel('Frame number')
        #ax[rows,cols].set_ylabel(column)
        ax[rows,cols].tick_params(axis='x', labelsize=14)
        ax[rows,cols].tick_params(axis='y', labelsize=14)
        ax[rows,cols].legend(fontsize = 14)

        print(rows,cols)

        if cols == 0:
            cols += 1

        elif rows < 3:
            rows += 1
            cols = 0


    #ax[0,0].set_xlabel('Frame number', fontsize = 18)
    ax[0,0].set_ylabel(r'Area [px]', fontsize = 18)
    #ax[0,0].set_title('Area', fontsize = 18)


    #ax[0,1].set_xlabel('Frame number', fontsize = 18)
    ax[0,1].set_ylabel('Diameter [px]', fontsize = 18)
    #ax[0,1].set_title('Diameter', fontsize = 18)

    #ax[1,0].set_xlabel('Aspect ratio', fontsize = 18)
    ax[1,0].set_ylabel('Bubble ratio', fontsize = 18)
    #ax[1,0].set_title('Aspect ratio', fontsize = 18)

    #ax[1,1].set_xlabel('Frame number', fontsize = 18)
    ax[1,1].set_ylabel('Solidity', fontsize = 18)
    #ax[1,1].set_title('Solidity', fontsize = 18)

    #ax[2,0].set_xlabel('Frame number', fontsize = 18)
    ax[2,0].set_ylabel('Orientation [°]', fontsize = 18)
    #ax[2,0].set_title('Orientation', fontsize = 18)

    #ax[2,1].set_xlabel('Frame number', fontsize = 18)
    ax[2,1].set_ylabel('Extent', fontsize = 18)
    #ax[2,1].set_title('Extent', fontsize = 18)

    ax[3,0].set_xlabel('Frame number', fontsize = 18)
    ax[3,0].set_ylabel('Perimeter [px]', fontsize = 18)
    #ax[3,0].set_title('Perimeter', fontsize = 18)

    ax[3,1].set_xlabel('Frame number', fontsize = 18)
    ax[3,1].set_ylabel('Roundness', fontsize = 18)
    #ax[3,1].set_title('Roundness', fontsize = 18)
        
    plt.tight_layout()    
    plt.savefig('./moving_average_individual_analysis.png')
    plt.show()
        