import pandas as pd
import numpy as np
from preprocessing import SaNGreeATransformer



data = {
    'age': [25, 30, 22, 28, 35],
    'gender_M': [1, 0, 1, 0, 0],  
    'gender_F': [0, 1, 0, 1, 1]
}

df_encoded = pd.DataFrame(data)
print("Original DataFrame:")
print(df_encoded)



anon = SaNGreeATransformer(k=2)

anon.fit(df_encoded, k_graph=2)  

df_anon = anon.transform(df_encoded)

print("\nAnonymized DataFrame:")
print(df_anon)