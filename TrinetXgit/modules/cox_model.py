#This is python function and procedures, what is called in the main notebook. 
#it contains: 

# 1) cox_model(X_train,X_test,data_type = 'training',penalizer=0.0001, l1_ratio=0.9)
#    General Cox model 1 

# 2) cox_feature_importancy(X_train, X_test, clean_feature = True ,data_type = 'training', feature_excluded_from_plot =[],size = (30,40), directory_name='',penalizer=0.0001, l1_ratio=0.9, additional_name='' ):
#    Cox model 2 with feature importancy plot.
#    This function allows to create the cox model with/without Lasso less significant features clean up method and build the plot of important features 
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import pickle
import json  
import matplotlib.font_manager  
import matplotlib.pyplot as plt
import seaborn as sns

# %% JSON file with feature names - in case of new features add new names inside
with open('../dictionaries/data_feature_names.json', 'r') as file:
    feature_mapping = json.load(file)

# General Cox model without Lasso feature clean up.
# Return back C-index for train and test sets. 
        
# for training dataset it creates the cox model data fit and save this model to  models/cox_model.pkl
# for test dataset it uses saved model from models/cox_model.pkl
           
def cox_model(X_train,X_test,data_type = 'training',penalizer=0.0001, l1_ratio=0.9):
    if data_type =='training':
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        cph.fit(X_train, duration_col="remainder__l2_TTNT", event_col="remainder__event")
        score_X_train = cph.score(X_train, scoring_method="concordance_index")
        score_X_test = cph.score(X_test, scoring_method="concordance_index")

        with open('../models/cox_model.pkl', 'wb') as file:
            pickle.dump(cph, file)
    elif data_type=='test':
        with open('../models/cox_model.pkl', 'rb') as file:
            cph = pickle.load(file)
            score_X_test = cph.score(X_test, scoring_method="concordance_index")
            coefficients = cph.summary
        score_X_train = 'not related'


    return score_X_train,score_X_test


# Cox model with feature importancy plot.
# This function allows to create the cox model with/without Lasso less significant features clean up method and build the plot of important features 

# clean_feature = True / False - cox model with/without Lasso less significant features clean up method
#data_type = 'training' /'test' - type of dataset
# feature_excluded_from_plot =[]  - specify the list of features, what you want to exclude from the plot construction
# size = (30,40) - choose size of the plot
# directory_name='' - current directory where all information about feature importancy and 'directory_name/figures' -  plot save 

#Specify the parameters of the cox model: 
#penalizer=0.0001, 
#l1_ratio=0.9, 

#additional_name='' - is used in the loops to save different models with different parameters

def cox_feature_importancy(X_train, X_test, clean_feature = True ,data_type = 'training', feature_excluded_from_plot =[],size = (30,40), directory_name='',penalizer=0.0001, l1_ratio=0.9, additional_name='' ):
    if data_type == 'training':
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

        # Significant feature clean up. This procedure drop less significant features
        if clean_feature == True:
            dropped_columns = []
            while True:
                
                cph.fit(X_train, duration_col="remainder__l2_TTNT", event_col="remainder__event")
                coefficients = cph.summary
                least_significant_features = coefficients[abs(coefficients["p"])>0.60].index.to_list()
                print(least_significant_features)
                print(X_train.shape)
                # print(cph.score(X_train, scoring_method="concordance_index"), cph.score(X_test, scoring_method="concordance_index"))
                X_train = X_train.drop(columns=least_significant_features)
                dropped_columns.extend(least_significant_features) 
                if not bool(least_significant_features):
                    break
            with open(f'../dictionaries/clean_feature_dropped_columns_{additional_name}.json', 'w') as file:
                json.dump(dropped_columns, file)

        cph.fit(X_train, duration_col="remainder__l2_TTNT", event_col="remainder__event")
        coefficients = cph.summary
        
        with open(f'../models/cox_model_feature_importancy_{additional_name}.pkl', 'wb') as file:
            pickle.dump(cph, file)
        coefficients = cph.summary

        score_X_train = cph.score(X_train, scoring_method="concordance_index")
        score_X_test = cph.score(X_test, scoring_method="concordance_index")
    
    elif data_type == 'test':
        if clean_feature == True:
            with open(f'../dictionaries/clean_feature_dropped_columns_{additional_name}.json', 'r') as file:
                dropped_columns = json.load(file)

            X_test = X_test.drop(columns=dropped_columns, errors='ignore')

        with open(f'../models/cox_model_feature_importancy_{additional_name}.pkl', 'rb') as file:
            cph = pickle.load(file)
            score_X_test = cph.score(X_test, scoring_method="concordance_index")
            coefficients = cph.summary
        score_X_train = 'not related'
        
    


    # sorted_features = coefficients.sort_values(by='coef', key=abs, ascending=False)

    sorted_features = coefficients[abs(coefficients['p'])>0.0]
    #All features with error intervals
    top_features = sorted_features.sort_values(by='coef', ascending=False) #.head(num_top_features).sort_values(by='coef', ascending=True)
    # top_features = sorted_features.head(num_top_features).sort_values(by='coef', ascending=False)
        
    protective = top_features[top_features['exp(coef)']<1][top_features['exp(coef) upper 95%']<1]
    detrimental = top_features[top_features['exp(coef)']>1][top_features['exp(coef) lower 95%']>1]
    top_features_selected = pd.concat([detrimental, protective])
    #top_features_selected = top_features
    

    top_f_df = top_features_selected[['exp(coef)', 'se(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].reset_index().rename(columns={'exp(coef)': 'HR', 'exp(coef) lower 95%': 'ci_low',  'exp(coef) upper 95%': 'ci_high', 'se(coef)': 'se'})
    top_f_df = top_f_df.sort_values(by='HR', ascending=False).reset_index(drop=True)

    x_error = np.vstack([top_f_df['HR'] - top_f_df['ci_low'], top_f_df['ci_high'] - top_f_df['HR']])

    top_f_df['ci_left'] = x_error[0]
    top_f_df['ci_right'] = x_error[1]

    top_f_df.to_csv(f"{directory_name}/feature_selection.csv")
    
    #selected features for the plot with condition error range >1 or <1
    top_f_df

    top_f_df["feature_name"] = top_f_df["covariate"].map(feature_mapping)
    print(top_f_df[['covariate','feature_name']])
    #print(top_f_df['covariate'])
    #print(top_f_df['feature_name'])
        #top_f_df["feature_name"] = top_f_df["covariate"]

        #style of text Kanit
    fonts = matplotlib.font_manager.findSystemFonts(fontpaths="../dictionaries/fonts/", fontext="ttf")
    for font_file in fonts:
        matplotlib.font_manager.fontManager.addfont(font_file)
        
    matplotlib.font_manager.findfont("Kanit")
    #plot creation

    sns.set(style="white", color_codes=True, palette='viridis')
    colors = ['red' if value > 1 else '#323e68' for value in top_f_df['HR']]
    fontsize = 50
    # plot_errorbars(lambda x: (x['ci_low'], x['ci_high']))
    plt.figure(figsize=size)
    plt.axvline(x=1, color='gray', linestyle='--')
    # Create a Seaborn point plot with error bars for confidence intervals
    # sns.pointplot(y="covariate", x="HR", data=top_f_df, errorbar=('ci', 95)) #((top_f_df['ci_low'], top_f_df['ci_high'])))
    i=0

    for feature, category, value, errorl,errorr in zip(top_f_df['covariate'],top_f_df['feature_name'], top_f_df['HR'], top_f_df['ci_left'], top_f_df['ci_right']):
        if feature not in feature_excluded_from_plot:
            color = 'red' if value > 1 else '#323e68'
            plt.errorbar(y=category, x=value, xerr=([errorl], [errorr]), fmt='o', capsize=1, 
                    linewidth=5.0, markersize=20,  ecolor=color, color=color, elinewidth=5.5)#sns.color_palette()[2])
            plt.text(value, category, category, ha='center', va='bottom', color='black', fontweight='normal', position=(value, i+0.2), size=fontsize, fontname="Kanit")
            i += 1

    plt.text(1, 0, 'High Risk \n Shorter Time', ha='center', va='bottom', color='black', fontweight='bold', position=(1.2, 0), size=fontsize, fontname="Kanit")
    plt.text(1, 0, 'Low Risk \n Longer Time', ha='center', va='bottom', color='black', fontweight='bold', position=(0.8, 0), size=fontsize, fontname="Kanit")
    # Set plot title and labels

    # plt.title('Features', fontsize=20, fontweight="bold")
    plt.xlabel('HR (CI 95%)',fontweight='bold', fontsize=fontsize, fontname="Kanit")
    plt.ylabel('Features',fontweight='bold', fontsize=fontsize, fontname="Kanit")
    plt.xticks(fontsize=fontsize, fontname="Kanit")
    plt.yticks([], fontsize=fontsize)
    # ax2 = plt.twinx()
    # ax2.set_yticks(top_f_df['HR'])
    # ax2.set_yticklabels(top_f_df['HR'])
    # plt.xlim(0.5, 2.3)
    
  


    plt.savefig(f"{directory_name}/figures/feature_selection_{additional_name}.png")
    # Show the plot
    # plt.gca().invert_xaxis()
    plt.gcf().set_facecolor('none')
    plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)
    plt.show()
    return top_f_df, score_X_train, score_X_test, cph 