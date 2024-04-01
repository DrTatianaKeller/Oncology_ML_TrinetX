import missingno as msno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_missing_values(df,directory_name):
    number_of_features = df.shape[1]
    
    # Group features into sets of 30
    feature_groups = [df.iloc[:, i:i + 30] for i in range(0, number_of_features, 30)]

    # Create a PDF file to save the plots
    pdf_filename = f'{directory_name}\figures\missing_values_plots.pdf'
    pdf_pages = []

    for group_index, feature_group in enumerate(feature_groups):
        fig, axes = plt.subplots(1, 1, figsize=(18, 14))
        msno.matrix(feature_group, ax=axes)
        plt.title(f'Missing Values - Features {group_index * 30 + 1} to {(group_index + 1) * 30}')

        # Add the plot to the PDF
        pdf_pages.append(plt.gcf())
        
    # Save all the plots to a single PDF file
    with PdfPages(pdf_filename) as pdf:
        for page in pdf_pages:
            pdf.savefig(page)
            
    print(f'Missing value plots saved to {pdf_filename}')