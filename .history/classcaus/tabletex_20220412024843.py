from cProfile import label
import pandas as pd
import numpy as np
# define a function taking pandas dataframe as input and returning a text table and latex table

def table_tex(df, title, caption='',path_data=''):
    """
    This function takes a pandas dataframe as input and returns  latex code.
    :param df: pandas dataframe
    :param title: title of the table
    :param caption: caption of the table
    :param label: label of the table
    :param path_data: path to the data
    :return: text table and latex table
    """
    label = title
    latex_table = df.to_latex(index=False)
    latex_table = latex_table.replace('toprule', '\\hline')
    latex_table = latex_table.replace('midrule', '\\hline')
    latex_table = latex_table.replace('bottomrule', '\\hline')
    latex_table = latex_table.replace('\\begin{tabular}{', '\\begin{table}[h]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\begin{tabular}{')
    latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\n\\end{table}')
   
    print(latex_table)
    # save the latex table
    with open(path_data + 'table_' + title + '.tex', 'w') as f:
        f.write(latex_table)
    return df.to_latex(index=False)


    