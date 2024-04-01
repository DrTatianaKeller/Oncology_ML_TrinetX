#This is python function and procedures, what is called in the main notebook. 
#it contains: 

# 1) datatype_extraction_procedure(df, path_list="dictionaries/", report = 0)
#    procedure that return back the lists of different types variable from the dataset

# 2) cox_model(X_train,X_test,data_type = 'training',penalizer=0.0001, l1_ratio=0.9)
#    General Cox model

# 3) cox_feature_importancy(X_train, X_test, clean_feature = True ,data_type = 'training', feature_excluded_from_plot =[],size = (30,40), directory_name='',penalizer=0.0001, l1_ratio=0.9, additional_name='' ):
#    Cox model with feature importancy plot.
#    This function allows to create the cox model with/without Lasso less significant features clean up method and build the plot of important features 

# 4) Kaplan_Meier_plot(X_train,lim_min_percent=10,feature_groups_specific=[], plot_name='', directory_name=''):
#    Kaplan Meier curves for balance categories

# 5) color_style()
#    color and style of plots     


# %% Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import CoxPHFitter

from sklearn.base import BaseEstimator, TransformerMixin, _SetOutputMixin

import matplotlib.font_manager
import pickle

from matplotlib.backends.backend_pdf import PdfPages

from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import warnings
from lifelines.utils import ApproximationWarning
import json    

# %% JSON file with feature names - in case of new features add new names inside
with open('dictionaries/data_feature_names.json', 'r') as file:
    feature_mapping = json.load(file)




# procedure that return back the lists of different types variable from the dataset:
# (num_features,bin_features,cat_features,ord_features,ddmmyy_features,all_cat_features)
    
#   if you want to control the datatype list in output: 
    # choose (report = 1 )
    
def datatype_extraction_procedure(df, path_list="dictionaries/", report = 0):
    df_data_type = pd.read_csv(path_list + "index_list_num_cat_ord_bin_corrected.csv")  
    #rename columns in df_data_type
    new_column_names = ["ind2", "feature", "data_type"]
    df_data_type.rename(columns=dict(zip(df_data_type.columns, new_column_names)), inplace=True)
    df_data_type=df_data_type.drop(columns = "ind2")

    def data_type_list(dataframe, *data_type_names):
    
        data_types = []
        for data_type_name in data_type_names:
            data_types.append(data_type_name)
        
        # Filter the DataFrame to select features with specified data types
        lib_features = df_data_type[df_data_type["data_type"].isin(data_types)]["feature"].tolist()

        features = []
        for feature in dataframe.columns:
            if feature in lib_features:
                features.append(feature)
        return features
    num_features = data_type_list(df, "num")
    bin_features = data_type_list(df, "bin")
    ddmmyy_features = data_type_list(df, "date")
    cat_features = data_type_list(df, "cat")
    ord_features = data_type_list(df, "ord")
    all_cat_features = data_type_list(df, "cat","ord","bin")

    if report == 1:  
        print("Control output of used datatype: ")
        
        print("numerical: ")
        print(num_features)

        print("bin: ")
        print(bin_features)
       
        print("date: ")
        print(ddmmyy_features)
        
        print("categorized: ")
        print(cat_features)
       
        print("ordered: ")
        print(ord_features)

        print("all cat.: ")
        print(all_cat_features)

    return (num_features,bin_features,cat_features,ord_features,ddmmyy_features,all_cat_features)
    



# %%
class Rounder(BaseEstimator, TransformerMixin, _SetOutputMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda x: np.ceil(x).astype(int))
    
    def set_output(self, *, transform=None):
        """Set output container.
        Parameters
        ----------
        transform : {"default", "pandas"}, default=None
            Configure output of `transform` and `fit_transform`.

            - `"default"`: Default output format of a transformer
            - `"pandas"`: DataFrame output
            - `None`: Transform configuration is unchanged

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if transform is None:
            return self

        if not hasattr(self, "_sklearn_output_config"):
            self._sklearn_output_config = {}

        self._sklearn_output_config["transform"] = transform
        return self

# %%
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

        with open('models/cox_model.pkl', 'wb') as file:
            pickle.dump(cph, file)
    elif data_type=='test':
        with open('models/cox_model.pkl', 'rb') as file:
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
            with open(f'dictionaries/clean_feature_dropped_columns_{additional_name}.json', 'w') as file:
                json.dump(dropped_columns, file)

        cph.fit(X_train, duration_col="remainder__l2_TTNT", event_col="remainder__event")
        coefficients = cph.summary
        
        with open(f'models/cox_model_feature_importancy_{additional_name}.pkl', 'wb') as file:
            pickle.dump(cph, file)
        coefficients = cph.summary

        score_X_train = cph.score(X_train, scoring_method="concordance_index")
        score_X_test = cph.score(X_test, scoring_method="concordance_index")
    
    elif data_type == 'test':
        if clean_feature == True:
            with open(f'dictionaries/clean_feature_dropped_columns_{additional_name}.json', 'r') as file:
                dropped_columns = json.load(file)

            X_test = X_test.drop(columns=dropped_columns, errors='ignore')

        with open(f'models/cox_model_feature_importancy_{additional_name}.pkl', 'rb') as file:
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
    fonts = matplotlib.font_manager.findSystemFonts(fontpaths="helper/fonts/", fontext="ttf")
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


# cox model with feature importancy - old version what works only with train data.
def cox_feature_importancy2(X_train, X_test, clean_feature = True , feature_excluded_from_plot =[],size = (30,40), directory_name='',penalizer=0.001, l1_ratio=0.9 ):

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

    # Significant feature clean up. This procedure drop less significant features
    if clean_feature == True:
        while True:
            
            cph.fit(X_train, duration_col="remainder__l2_TTNT", event_col="remainder__event")
            coefficients = cph.summary
            least_significant_features = coefficients[abs(coefficients["p"])>0.60].index.to_list()
            print(least_significant_features)
            print(X_train.shape)
            # print(cph.score(X_train, scoring_method="concordance_index"), cph.score(X_test, scoring_method="concordance_index"))
            X_train = X_train.drop(columns=least_significant_features)
            if not bool(least_significant_features):
                break
    cph.fit(X_train, duration_col="remainder__l2_TTNT", event_col="remainder__event")
    coefficients = cph.summary

    score_X_train = cph.score(X_train, scoring_method="concordance_index")
    score_X_test = cph.score(X_test, scoring_method="concordance_index")

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
    fonts = matplotlib.font_manager.findSystemFonts(fontpaths="helper/fonts/", fontext="ttf")
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
    
  


    plt.savefig(f"{directory_name}/figures/feature_selection.png")
    # Show the plot
    # plt.gca().invert_xaxis()
    plt.gcf().set_facecolor('none')
    plt.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)
    plt.show()
    return top_f_df, score_X_train, score_X_test

#color and style of plots     
def color_style():
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


# %%
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




