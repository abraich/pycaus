import pandas as pd
import numpy as np
# define a function taking pandas dataframe as input and returning a text table and latex table

def table_tex(df, title, caption, label, path_data):
    """
    This function takes a pandas dataframe as input and returns  latex code.
    :param df: pandas dataframe
    :param title: title of the table
    :param caption: caption of the table
    :param label: label of the table
    :param path_data: path to the data
    :return: text table and latex table
    """
    
    