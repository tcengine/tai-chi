import re

def clean_name(x:str)->str:
    """
    Clean the pandas dataframe name to be used as a file name
    """
    return re.sub(r'[^\w\s]', '', x)