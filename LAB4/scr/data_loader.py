import numpy as np 
import pandas as pd
import os

def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    red_link = os.path.join(data_dir, 'winequality-red.csv')
    white_link = os.path.join(data_dir, 'winequality-white.csv')

    red_df = pd.read_csv(red_link, sep=';')
    white_df = pd.read_csv(white_link, sep=';')

    red_df['class'] = 1
    white_df['class'] = 0
    df = pd.concat([red_df, white_df], axis=0)
    data = df.values
    return data

