
#comparison plot for different imputers for the same data type
import matplotlib.pyplot as plt

def plot_imputer_performance_comparison(Ylim_1, Ylim_2, fontsize, dictionary_imputer, additional_name_plot,imputers_type,imputer_choice_directory):

    # Extracting data for plotting
    imputer_keys = []
    train_scores = []
    test_scores_train_data = []
    test_scores_test_data = []

    for imputer_key, data in dictionary_imputer.items():
        # Splitting the imputer name and joining with a newline
        split_name = '\n'.join(imputer_key.replace(')', '').split('('))
        imputer_keys.append(split_name)
        
        train_scores.append(data['training']['score_X_train'])
        test_scores_train_data.append(data['training']['score_X_test'])
        
        # Check if 'test' data exists for the imputer
        if 'test' in data:
            test_scores_test_data.append(data['test']['score_X_test'])
        else:
            test_scores_test_data.append(None)  # Or a placeholder value like 0 or NaN

    # Plotting
    plt.figure(figsize=(23, 12))

    # Adding data to the plot
    plt.plot(imputer_keys, train_scores, marker='o', label='Train Score (Training Data)')
    plt.plot(imputer_keys, test_scores_train_data, marker='o', label='Test Score (Training Data)')
    plt.plot(imputer_keys, test_scores_test_data, marker='o', label='Test Score (Test Data)')

    # Customizing the plot
    plt.xlabel('Imputer Type',fontsize=fontsize)
    plt.ylabel('C-index',fontsize=fontsize)
    plt.title(f'{imputers_type} Imputer Performance Comparison',fontsize=fontsize+8)
    #plt.ylim(0.45, 0.7)  # Setting y-axis limits
    plt.ylim(Ylim_1, Ylim_2)  # Setting y-axis limits

    plt.xticks(rotation=45,fontsize=fontsize)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=fontsize)  # Adjust the font size as needed
    plt.legend(fontsize=fontsize)

    # Show plot
    plt.tight_layout()
    plt.savefig(f'{imputer_choice_directory}/figures/{imputers_type}_data_imputer_{additional_name_plot}.png')
    plt.show()
