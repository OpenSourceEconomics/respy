#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import pandas as pd
import numpy as np

columns = ['Identifier', 'Age', 'Schooling', 'Choice', 'Wage']
dtype = {'Identifier': np.int, 'Age': np.int,  'Schooling': np.int,  'Choice': 'category'}
df_kw = pd.DataFrame(np.genfromtxt('KW_97.raw'), columns=columns).astype(dtype)
df_kw.set_index(['Identifier', 'Age'], inplace=True, drop=False)
df_kw['Choice'].cat.categories = ['Schooling', 'Home', 'White', 'Blue', 'Military']

with open('career_decisions_data.respy.dat', 'w') as file_:
    df_kw.to_string(file_, index=False, header=True, na_rep='.')