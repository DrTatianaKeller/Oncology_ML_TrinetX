import re
def name_imputer_extract(imputer):
    phrase = f'{imputer}'
    # Splitting the phrase by non-alphanumeric characters to isolate words and numbers
    parts = [part for part in re.split(r'\W+', phrase) if part]
    name = ''
    for item in parts:
        name = name+'_'+item
    
    return name