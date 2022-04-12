import texttable 
import pandas as pd
import numpy as np
# define a function taking pandas dataframe as input and returning a text table and latex table

def table_tex(df, title, caption, label, path_data):
    """
    This function takes a pandas dataframe as input and returns a text table and latex table.
    :param df: pandas dataframe
    :param title: title of the table
    :param caption: caption of the table
    :param label: label of the table
    :param path_data: path to the data
    :return: text table and latex table
    """
    # create a text table
    table = texttable.Texttable()
    table.header(df.columns)
    table.add_rows(df.values)
    table_str = table.draw()
    # create a latex table
    latex_table = table_str.replace('\\', '\\\\')
    latex_table = latex_table.replace('|', '\\|')
    latex_table = latex_table.replace('-', '\\-')
    latex_table = latex_table.replace('_', '\\_')
    latex_table = latex_table.replace('^', '\\^')
    latex_table = latex_table.replace('~', '\\~')
    latex_table = latex_table.replace('&', '\\&')
    latex_table = latex_table.replace('%', '\\%')
    latex_table = latex_table.replace('$', '\\$')
    latex_table = latex_table.replace('#', '\\#')
    latex_table = latex_table.replace('{', '\\{')
    latex_table = latex_table.replace('}', '\\}')
    latex_table = latex_table.replace('\n', '\n\n')
    
    print(latex_table)
    # save the latex table
    with open(path_data + 'table_' + label + '.tex', 'w') as f:
        f.write(latex_table)
    # save the text table
    with open(path_data + 'table_' + label + '.txt', 'w') as f:
        f.write(table_str)
    # return the text table and latex table
    return table_str, latex_table

if __name__ == '__main__':
    # create a pandas dataframe
    df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
    # create a text table
    table_str = table_tex(df, 'My table', 'My caption', 'My label', './')
    # create a latex table
    latex_table = table_tex(df, 'My table', 'My caption', 'My label', './')
    
    