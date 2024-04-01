# %%
#This is python function, what is called in the main notebook. 
#it contains: 
# 1) Kaplan_Meier_plot(X_train,lim_min_percent=10,feature_groups_specific=[], plot_name='', directory_name=''):
#    Kaplan Meier curves for balance categories    

import numpy as np
import pandas as pd
import warnings
from lifelines.utils import ApproximationWarning
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import json    


# %% JSON file with feature names - in case of new features add new names inside
with open('../dictionaries/data_feature_names.json', 'r') as file:
    feature_mapping = json.load(file)

#Kaplan Meier curves for balance categories
    #X_train - data for plot construction
    #lim_min_percent=10 - min percent of the category. the category less then this percent will not be shown on the plot
     
    #feature_groups_specific=[] - specify the list of the features, what should be shown on the save plots. 
    #plot_name='' - personalise the name of the plot
    #directory_name='' - current directory where all plots are saved

def Kaplan_Meier_plot(X_train,lim_min_percent=10,feature_groups_specific=[], plot_name='', directory_name=''):
    
    # Ignore the ApproximationWarning
    plt.style.use('seaborn')
    warnings.simplefilter("ignore", category=ApproximationWarning)

    color_palette_lib = {
        'vibrant_cyan': '#43aecc',
        'neutral_grey': '#78787c',
        'deep_blue': '#323e68',
        'darker_grey': '#5b5b60',
        'darker_blue': '#4a5572',
        'lighter_cyan': '#6bc7e5',
        'muted_greenish_blue': '#526f7b',
        'soft_purple': '#867c9a',
        'muted_burgundy_red': '#8B3A3A'  # The added deep muted red
    }

    color_palette = [
    
        '#323e68',  # Original deep blue
        '#8B3A3A',   # Deep muted red (added)
        '#43aecc',  # Original vibrant cyan
        '#78787c',  # Original neutral grey
        
        '#5b5b60',  # Darker grey
        '#4a5572',  # Darker blue
        '#6bc7e5',  # Lighter cyan
        '#526f7b',  # Muted greenish-blue
        '#867c9a',  # Soft purple
        
    ]

    feature_groups = feature_groups_specific


    percent_of_ones = {}
    for feature in X_train.columns:
        
        if 'bin_' in feature or 'cat_' in feature:
            percent_of_ones[feature] = (X_train[feature] == 1).mean() * 100




    min_group_percent={}
    exclude_data=[]
    i=0
    min_percent = 100

    for feature in feature_groups_specific:
        if feature in X_train.columns:
            if percent_of_ones[feature]< min_percent and percent_of_ones[feature]>lim_min_percent:
                min_percent = percent_of_ones[feature]
            elif percent_of_ones[feature]<lim_min_percent:
                exclude_data.append(feature)
        else: 
            print(feature + ' not exist in X_train') 
    #print(min_percent)
    min_group_percent = min_percent

    print(min_group_percent)
    print(exclude_data)


    sns.set(style="white", color_codes=True, palette='viridis')
    with PdfPages(f'{directory_name}/figures/Kaplan_Meier_minpercent_selected_{lim_min_percent}_{plot_name}.pdf') as pdf:

        # Sample data (time to event and event indicator)
        durations = X_train['remainder__l2_TTNT']  # Time to event


        # Plot Kaplan-Meier for each group
        plt.figure(figsize=(8, 6))
        kmf = KaplanMeierFitter()
        i=0
        for feature in feature_groups:
            count = 0
        
            if feature not in exclude_data and feature in X_train.columns:
                #if feature not in exclude_data:
                    #sample_data = X_train[X_train[feature] == 1][feature].sample(int(min_group_percent[group]*len(X_train)/100))
                sample_data = X_train[X_train[feature] == 1].sample(int(min_group_percent*len(X_train)/100))
                sample_category = sample_data[feature]
                sample_durations = sample_data['remainder__l2_TTNT']
                kmf.fit(sample_durations, sample_category, label=feature_mapping[feature])
                kmf.plot_survival_function(color=color_palette[i])

                line_width = 0.7
                # Find the x-value where survival is 0.5
                time_at_50_percent_survival = kmf.percentile(0.5, )
                print(f'{feature}: {time_at_50_percent_survival}')

                time_at_50_percent_survival_difference = time_at_50_percent_survival-time_at_50_percent_survival

                # Draw a vertical line from the intersection point to the x-axis
                plt.plot([time_at_50_percent_survival, time_at_50_percent_survival], [0, 0.5], color='black', linestyle='--', linewidth=line_width)
                
                # Draw a horizontal line from the intersection point to the y-axis
                plt.plot([0, time_at_50_percent_survival], [0.5, 0.5], color='black', linestyle='--', linewidth=line_width)
            i=i+1


        plt.title(f'Kaplan-Meier Curves',fontsize=20,fontname="Kanit")
        plt.xlabel('Days',fontsize=18,fontname="Kanit")
        plt.ylabel('patients not yet undergone next treatment',fontsize=18,fontname="Kanit")
        plt.xticks(fontsize=18,fontname="Kanit")
        plt.yticks(fontsize=18,fontname="Kanit")
        plt.legend(fontsize=14)
        #plt.yscale('log')
        #plt.xlim(1000)
        plt.xlim(0, 1000)
        plt.gcf().set_facecolor('none')
        plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)

        pdf.savefig()
        plt.show()
        # count += 1
        # if count == 1:
        #     break