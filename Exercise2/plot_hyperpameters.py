import matplotlib.pyplot as plt
import numpy as np 


ford_dict = {'n_estimators=150': [1, 17.02, 1.34], 'n_estimators=100': [2, 17.06, 1.33], 'n_estimators=80': [3, 17.06, 1.31], 'n_estimators=50': [4, 17.12, 1.36],
               'max_depth=50': [5, 17.06, 1.37], 'max_depth=30': [6, 17.06, 1.33], 'max_depth=20': [7, 16.59, 1.26], 'max_depth=None': [8,17.06, 1.33],
               'min_samples_split=15': [9, 15.47, 1.14], 'min_samples_split=10': [10, 15.82, 1.19], 'min_samples_split=5': [11, 16.54,  1.27], 'min_samples_split=2': [12, 17.06, 1.33]}

house_dict = {'n_estimators=150': [1, 821504.5, 52927.3], 'n_estimators=100': [2, 820154.5, 51798.3], 'n_estimators=80': [3, 819660, 50551.7], 'n_estimators=50': [4, 825761.5, 45405.8],
               'max_depth=50': [5, 820154.5, 51798.3], 'max_depth=30': [6, 820195.0,51826.7], 'max_depth=20': [7, 819047.0, 51627.6], 'max_depth=None': [8, 820154.5, 51798.3],
               'min_samples_split=15': [9, 808254.1, 48170.5], 'min_samples_split=10': [10, 813438.5, 49634.7], 'min_samples_split=5': [11, 818869.2,  53088.3], 'min_samples_split=2': [12, 820154.5,51798.3]}


teen_dict = {'n_estimators=150': [1, 0.39, 0.05], 'n_estimators=100': [2, 0.39, 0.05], 'n_estimators=80': [3, 0.39, 0.05], 'n_estimators=50': [4, 0.40, 0.06],
               'max_depth=50': [5, 0.39, 0.05], 'max_depth=30': [6, 0.39, 0.05], 'max_depth=20': [7, 0.39, 0.05], 'max_depth=None': [8, 0.39, 0.05],
               'min_samples_split=15': [9, 0.41, 0.05], 'min_samples_split=10': [10, 0.4, 0.05], 'min_samples_split=5': [11, 0.39, 0.052], 'min_samples_split=2': [12, 0.39, 0.05]}



for d, c in zip([ford_dict, house_dict, teen_dict], ['green', 'red', 'blue']):

  xs, ys, yerrs, labels = [], [], [], []
  for key, values in d.items():
     xs.append(values[0])
     ys.append(values[1])
     yerrs.append(values[2])
     labels.append(key)

  xs = np.array(xs)
  ys = np.array(ys)
  yerrs = np.array(yerrs)

  order = np.argsort(xs)
  xs, ys, yerrs = xs[order], ys[order], yerrs[order]


  plt.errorbar(xs, ys, yerr=yerrs, fmt='o-', linewidth=2, linestyle='dashed', capsize=6, color=c) 


  for x, y, yerr, label in zip(xs, ys, yerrs, np.array(labels)[order]):
    plt.annotate(
        label,
        (x, y + yerr),
        textcoords="offset points",
        xytext=(0, 5),
        ha='center',
        fontsize=9, rotation=45
    )

  plt.xlabel('Run', fontsize=12)
  if c=='red' or c =='green':
    plt.ylabel('MSE (*10^5)', fontsize=12) 
  else:
    plt.ylabel('MSE', fontsize=12) 
  plt.tight_layout()
  plt.show()