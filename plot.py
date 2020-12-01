import matplotlib.pyplot as plt
import numpy as np

def scatter(x, y, filename='Scatter.png', markersize=7, markeredgewidth=0.5, markeredgecolor='k', xlabel='Reference', ylabel='Prediction'):
    
    rmse = np.linalg.norm(x -y) / y.size**0.5
    mae = abs(x - y).mean()
    mini = min([min(x), min(y)])
    maxi = max([max(x), max(y)])
    
    plt.plot([mini, maxi], [mini, maxi], 'k-')
    plt.plot(x, y, 'o', markersize=markersize, markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('RMSE: %.3f   MAE: %.3f' % (rmse, mae))
    plt.show()
