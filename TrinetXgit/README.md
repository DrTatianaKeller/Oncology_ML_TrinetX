# Time to Next Treatment Prediction in Cancer Patients, Tackling missing data with machine learning
 
## Project Intro/Objective
The project is provided by [CancerDataNet](https://www.linkedin.com/company/cancerdatanet/posts/?feedView=all), developing a prognostic model for treatment outcomes and showing prognostic factors in oncology sorted according to their significance.
 
 
### Authors
The original project was started by team and then upgraded, structurised and automatized by Dr. Tatiana Keller.

The team:
* [Dr. Tatiana Keller](https://www.linkedin.com/in/dr-tatiana-keller/)
* [Dr. Antonio Mariano](https://www.linkedin.com/in/mariano-antonio/)
* [Gordon Marshall](https://www.linkedin.com/in/gordon-w-marshall/)
 
 
## Project Goal
 
In the project, the target is to identify key variables influencing patient Time To Next Treatment (TTNT) by leveraging Machine Learning and Real World Data.


## Project Description

Over 160,000 people worldwide are currently living with multiple myeloma - rare cancer that affects the production of plasma cells and causes a multitude of symptoms, including reduced kidney function, bone lesions, and anemia. 

How can we ensure timely and effective treatment for these patients? What are the key indicators that signify the success of the treatment?

TriNetX is a global health research network that provides a platform for healthcare organizations, researchers, and life sciences companies to collaborate and access real-world clinical data (RWD) for clinical research and analysis. TriNetX aims to accelerate clinical trials, improve study design, and enhance patient recruitment by offering a comprehensive and standardized view of patient populations across various healthcare institutions. Its key features include a data network for sharing patient data, analytics tools for population analysis, protocol optimization for clinical trials, streamlined patient recruitment, and the generation of real-world evidence from routine clinical data.

The team was provided with an anonymized RWD database of Multiple Myeloma patient treatment histories.

The data consisted of 390 attributes of 2600 observations (patients), each one linked with the Time To Next Treatment (TTNT), which is the time elapsed between the first and second line of treatment of a patient.

This database contained a high amount of incorrect, inaccurate, and missing data. Therefore, as a first necessary step, the team developed a data pipeline which would correct the inadequate data and impute the missing values.

Later Tatiana Keller created the analyser for imputer choice to improve the C-index score and choose the best imputer method. 

Next, the team faced the problem of understanding which attributes or features influence the TTNT. Since this time marks the occurrence of an event happening (the next treatment phase), the issue falls into the “Survival Analysis” category.
The approach followed, typical of the Survival Analysis, was to use a Cox Regression Model and evaluate the performance using its Concordance Index (C-index) in a test-train validation split. The C-index quantifies the ability of the model to correctly order pairs of subjects with respect to their survival times. A C-index of 0.5 indicates no discrimination (equivalent to random chance), while a C-index of 1.0 signifies perfect discrimination. In our case, the score obtained was 0.62, which indicates good discrimination in line with similar problems in the literature.
 
The team therefore leveraged this model to obtain a list of features reported in order of their Hazard Ratio (HR), meaning the relative risk of an event happening, which, to recall, in this case, is the following treatment line. Features with high HR contribute to shortening the time a patient needs for a new treatment phase, while features with low HR contribute to a longer time.

One can see the feature contribution in detail from the Kaplan-Meier Survival Curves. Patient performance ability (ECOG) is a crucial factor that influences the Time To Next Treatment (TTNT). Limited Self-Care Ability or Disabeleness short time in several months in comparison with ambulatory or restricted activities Patients. The model can also predict the effectiveness of different drugs; for more detailed analysis model needs to have patient stratification by age, cancer stadium, or survival ranking by C-index. That is a plan for future model improvement.


## Data
 
The real-world blood cancer dataset consisted of 390 attributes of 2600 observations (patients), each one linked with the Time To Next Treatment (TTNT), which is the time elapsed between the first and second line of treatment of a patient.
 
## Requirements
The Environment file contains all necessary python packages.
 
### Methods Used
* Machine Learning (Cox Regression Model)
* Evaluation of Concordance Index (C-index)
* Hazard Ratio (HR), meaning the relative risk of an event happening.
* Kaplan-Meier Survival Curves (feature contribution in detail)
* Data Visualization

 
## Results

* [Final upgraded presentation]( https://docs.google.com/presentation/d/1eN6pekqF14yXJp1f5t8FXSHmnx-FcsTEidnf9wTCjJc/edit?usp=sharing)


## Code Structure
### Main Scripts:
- `scripts/Analysis.ipynb`: Preprocesses data, creates a Cox model, and evaluates feature importance.
- `scripts/Analysis_imputer_choice.ipynb`: Assists in selecting the best imputer and scaling method for survival model analysis.

### Modules:
- `modules.display_set_up`: Sets up notebook display outputs.
- `modules.read_data`: Reads dataframes and saves data information.
- `modules.datatype_extraction`: Extracts and lists data types from the dataset.
- `modules.create_directory`: Manages directory creation for data storage.
- `modules.correction`: Corrects and cleans dataset values.
- `modules.mapping`: Maps categorical values to numerical values.
- `modules.feature_dropping`: Identifies and drops irrelevant or redundant features.
- `modules.data_preprocess`: Prepares data for machine learning models.
- `modules.cox_model`: Contains functions for Cox model analysis.
- `modules.Kaplan_Meier_plot`: Generates Kaplan-Meier curves for data analysis.
- Additional modules for color styling, imputer naming, and performance comparison plots.

## Directories
Includes directories for storing intermediate reports, data dictionaries, and processed data for analysis.

 ## Modules structure:


* **`modules.datatype_extraction`**: 

  Extract features datatype and create datatype lists from index_list_num_cat_ord_bin.csv

  - data type extraction function for one type of data: 
          *datatype_solo_extraction(df, type_name, path_list="dictionaries/")*

  - data type extraction function for all types of data:
          *datatype_extraction_procedure(df, path_list="dictionaries/", report = 0)*

* **`modules.create_directory`**: 
    - create new_directory in  parent_directory
    - create directory in /intermediate_report/ in format yy-mm-dd _ data_name to store all necessary information about dataset.

* **`modules.correction`**: 

  Data correction:

  1. Nein to Nan for ['dx_lightrat', 'l1_lightrat', 'l2_lightrat', 'dx_lightkap', 'l1_lightkap', 'l2_lightkap', 'dx_lightlam', 'l1_lightlam', 'l2_lightlam']
  2. Convert numerical features type to float (original type: object)
  3. correction for 'kofscore':  20.21 to Nan
  4. ECOG value 9.0 to Nan , bin category 0,1 - 0; 2,3,4 - 1
  5. dx_sex to dx_female_sex 1/0,  bin category
  6. Find outliers. Change for outliers 99-type input to Nan 
  7. BMI(Body mass index):  
      1.Find BMI and using possible BMI interval estimate the swap weight and height columns
      2. BMI for dx, l1, l2 instead of height and weight features
      3. Calculate relative BMI change : l1_BMI_change, l2_BMI_change
  8. Data type columns:
      1. Calculate duration of rtx
      2. Calculate relative age when patient was diagnosed
      3. Drop data ddmmyy type columns

* **`modules.mapping`**:

  Data mapping:

    1. Replace 'Ja'/ 'Nein'/ 'Nicht getestet' to 1/ 0/ NaN (99) numerical value
    2. Mapping object type data:
        1. Sorting by number in the line:
            Example: 
            '<3,5 mg/L': 1, '3,5 - 5,5 mg/L': 2, '>5,5 mg/L': 3
        2. Sorting by Stadium I, II, III: giving indexes of mapping 1, 2, 3
        3. For Unbekant Stadium: index 0

* **`modules.feature_dropping`**: 

  Dropping Features:
      
  **`'training'`**:

    1. One unique value in column - 54 feautures was dropped
    2. Imbalance feature drop and categories combination with small statistic (< 6%) -> Create list of combined categories per each feature
        - Save data about categories combination to 'dictionaries/replaced_values.json'
    3. Drop all repeated/ dublicate information/ high correlated values:
        - drop_list = ['dx_weight', 'l1_weight', 'l2_weight', 'dx_height', 'l1_height','l2_height',	'l1_BMI','l2_BMI', 'l1_klinstudie','l2_klinstudie','dx_mikrogl','l1_mikrogl','l2_mikrogl','dx_albugem','l1_albugem','l2_albugem','l1_protg']
        - Drop 'l1_protg' due to hight correlation with 'l2_protg'
    4. Drop outliers for ['dx_protg', 'l1_protg', 'l2_protg'] 

    5. After all steps remain features list is saved in  'dictionaries/features_list_to_keep.json'`

  **`'test'`**:    

    1. Open:
    'dictionaries/replaced_values.json'
        - to combine categories inside each feature the same way as was done for training dataset 
    'dictionaries/features_list_to_keep.json'
        - to drop all features the same way as for training dataset
        -if some column is missing, make warning to retrain dataset

    2. Remove outliers

* **`modules.data_preprocess`**: 

  1. Preprocess data
  
      **`'test'`**:
      
      *preprocess_test_data(df_load,additional_name='')* - to apply the same transformation that was learned from the training data
      
      **`'training'`**:
      
      fix choice of imputer preprocess:

        *preprocess_data(df_load, threshold_feature_drop=27,directory_name='models',additional_name='')*

        - drop features with high missing values: threshold_feature_drop=27
        - fixed imputers for different features categories:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            num_imputer = KNNImputer(n_neighbors=10)
            cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
            scaler = QuantileTransformer()
        - split the data into training and testing sets
        -return:
            X_train = preprocessor.fit_transform(X_train) # model can learn the parameters of the transformations from this data
            X_test= preprocessor.transform(X_test) #to apply the same transformation that was learned from the training data

      * function to choose imputers for each feature categories
      
  2. Data Preprocess with different imputer choice:

        *preprocess_data_choose_imputer(df_load, threshold_feature_drop=25,directory_name='models',additional_name='',cat_imputer=SimpleImputer(strategy='most_frequent'), num_imputer=KNNImputer(n_neighbors=10), scaler=QuantileTransformer())*
        
        - function to choose imputers for each feature categories
        - choose parameters values: 
            categorical and binomial (yes/no) feature imputer:
                cat_imputer=SimpleImputer(strategy='most_frequent'), 
            numerical feature imputer and numerical data scaler:
                num_imputer=KNNImputer(n_neighbors=10), 
                scaler=QuantileTransformer()

        - split the data into training (80%) and testing (20%) sets
        -return:
            X_train = preprocessor.fit_transform(X_train) # model can learn the parameters of the transformations from this data
            X_test= preprocessor.transform(X_test) #to apply the same transformation that was learned from the training data



* **`modules.cox_model`**:

  Cox model functions: 

  1) *cox_model(X_train,X_test,data_type = 'training',penalizer=0.0001, l1_ratio=0.9)*
      General Cox model 1 

  2) *cox_feature_importancy(X_train, X_test, clean_feature = True ,data_type = 'training', feature_excluded_from_plot =[],size = (30,40), directory_name='',penalizer=0.0001, l1_ratio=0.9, additional_name='' )*:
      Cox model 2 with feature importancy plot.
      This function allows to create the cox model with/without Lasso less significant features clean up method and build the plot of important features 


* **`modules.Kaplan_Meier_plot`**: 
  
  Kaplan Meier curves for balance categories:  
  *Kaplan_Meier_plot(X_train,lim_min_percent=10,feature_groups_specific=[], plot_name='', directory_name='')*
    

### Directories:
* **`intermediate_report`** 
  
  store all information and plot about current dataframe 

* **`dictionaries`**:

  *dictionaries/index_list_num_cat_ord_bin_corrected.csv* 
  
    - ! Add information about datatype in that file if you have new feature in the dataframe
  
  *dictionaries/data_feature_names.json*
   
    - dictionary of names for features in dataframe
  
  *dictionaries/replaced_values.json* 
  
    - features categories with percent amount less then 5% (or the other choosen one in function) what were combine together to new category (10).  
* **`data`:**
  
  - `data.raw` - place your training and test dataset here

  - `data.preprocess` - saved data after clean up process 


## Contact:
in case of question, please contact the author: dr.tatiana.keller@gmail.com Tatiana Keller








---
