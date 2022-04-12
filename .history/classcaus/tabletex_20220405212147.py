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
    latex_table = df.to_latex(index=False)
    latex_table = latex_table.replace('toprule', '\\hline')
    latex_table = latex_table.replace('midrule', '\\hline')
    latex_table = latex_table.replace('bottomrule', '\\hline')
    latex_table = latex_table.replace('\\midrule', '\\hline')
    latex_table = latex_table.replace('\\toprule', '\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline')
    latex_table = latex_table.replace('\\begin{tabular}{', '\\begin{table}[h]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\begin{tabular}{')
    latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\n\\end{table}')
    latex_table = latex_table.replace('\\begin{tabular}{', '\\begin{table}[h]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\begin{tabular}{')
    latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\n\\end{table}')
    latex_table = latex_table.replace('\\begin{tabular}{', '\\begin{table}[h]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\begin{tabular}{')
    latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\n\\end{table}')
    latex_table = latex_table.replace('\\begin{tabular}{', '\\begin{table}[h]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\begin{tabular}{')
    latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\n\\end{table}')
    
    print(latex_table)
    # save the latex table
    with open(path_data + 'table_' + label + '.tex', 'w') as f:
        f.write(latex_table)
    return df.to_latex(index=False)

if __name__ == '__main__':
    # define a pandas dataframe
    df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'b', 'c', 'd', 'e'], columns=['one', 'two', 'three'])
    # define the title, caption and label of the table
    title = 'My table'
    caption = 'My caption'
    label = 'my_label'
    # define the path to the data
    path_data = './'
    # call the function
    table_tex(df, title, caption, label, path_data)