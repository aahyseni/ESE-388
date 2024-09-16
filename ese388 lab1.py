# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import seaborn as sns


random_array = np.random.randn(100, 1)

random_series = pd.Series(random_array.flatten())

random_array2 = np.random.randn(100,3)


panda_df = pd.DataFrame(random_array2, columns=['X1', 'X2', 'X3'])

sns.pairplot(data=panda_df)