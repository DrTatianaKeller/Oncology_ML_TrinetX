import os
from datetime import datetime

#create new_directory in  parent_directory
def create_directory_internal(parent_directory, new_directory):
    new_directory_path = f'{parent_directory}/{new_directory}'
    if os.path.exists(new_directory_path):
        print(f'Directory {new_directory_path} exist')
    else:    
        os.makedirs(new_directory_path, exist_ok=True)
        print(f"Directory '{new_directory_path}' created successfully.")


# create directory in /intermediate_report/ in format yy-mm-dd _ data_name
# to store all necessary information about dataset. 
                
def create_directory_main(data_name, parent_directory = "intermediate_report"):
 # Get the current date and time
    now = datetime.now()
    parent_directory = f'../{parent_directory}'
    # Format the date and time as mm:dd:hh:mm
    date_time_str = now.strftime("%y-%m-%d")
    dir_name = f'{date_time_str}_{data_name}'
    # Name of the parent directory
    
    # Create the parent directory if it doesn't exist


    if os.path.exists(f'{parent_directory}'):
        print(f'Directory {parent_directory} exist')
    else:    
        os.makedirs(parent_directory, exist_ok=True)
        print(f"Directory '{parent_directory}' created successfully.")


    # Create a directory with the formatted name inside the parent directory
    directory_name = os.path.join(parent_directory, dir_name)
    if os.path.exists(directory_name):
        print(f'Directory {directory_name} exist')
    else:    
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    
    figures_directory_name = os.path.join(parent_directory, dir_name,'figures')
    if os.path.exists(figures_directory_name):
        print(f'Directory {figures_directory_name} exist')
    else:    
        os.makedirs(figures_directory_name)
        print(f"Directory '{figures_directory_name}' created successfully.")
            
    return directory_name