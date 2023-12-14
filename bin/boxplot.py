import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def main():

    directory_path = r'C:\Users\pierr\Desktop\Cours\BME3\Medical_Image_Analysis_Lab\MIALab_Fabien\bin\mia-result'

    result_name = '2023-12-14-14-58-39'
    file_path = os.path.join(directory_path, result_name, 'results.csv')

    df = pd.read_csv(file_path, delimiter=';')

    dice_jaccard_metrics = ['DICE', 'JACRD']
    hausdorff_metrics = ['HDRFDST', 'HDRFDST95']

    # Create separate figures for Dice/Jaccard and Hausdorff metrics
    fig_dice_jaccard, axs_dice_jaccard = plt.subplots(1, 2, figsize=(12, 6))
    fig_hausdorff, axs_hausdorff = plt.subplots(1, 2, figsize=(12, 6))

    # Adjust the space between subplots
    plt.subplots_adjust(top=0.88, hspace=0.3)  # Increased vertical space

    # Plot the result titles
    fig_dice_jaccard.suptitle(f'Result: {result_name}', fontsize=16, ha='center')
    fig_hausdorff.suptitle(f'Result: {result_name}', fontsize=16, ha='center')

    # Custom x-axis labels
    custom_labels = ['Amygdala', 'Grey', 'Hippo', 'Thalamus', 'White']

    # Plot Dice and Jaccard coefficients
    for i, metric_name in enumerate(dice_jaccard_metrics, 0):
        axs_dice_jaccard[i].set_title(f'Dice per Label' if metric_name == 'DICE' else f'Jaccard per Label')
        axs_dice_jaccard[i].set_xlabel('')  # Set x-axis label to an empty string
        axs_dice_jaccard[i].set_ylabel(f'Dice coefficient' if metric_name == 'DICE' else f'Jaccard coefficient')

        if metric_name in df.columns:
            data = [df[df['LABEL'] == label][metric_name] for label in df['LABEL'].unique()]
            axs_dice_jaccard[i].set_ylim(0, 1)  # Set y-axis limits for Dice and Jaccard plots
            axs_dice_jaccard[i].boxplot(data, labels=custom_labels)
        else:
            axs_dice_jaccard[i].text(0.5, 0.5, f'Metric {metric_name} not available', ha='center', va='center', fontsize=12,
                                   color='red')

    # Plot Hausdorff and 95th Hausdorff coefficients
    for i, metric_name in enumerate(hausdorff_metrics, 0):
        axs_hausdorff[i].set_title(f'Hausdorff Distance per Label' if metric_name == 'HDRFDST' else f'Hausdorff distance 95th per Label')
        axs_hausdorff[i].set_xlabel('')  # Set x-axis label to an empty string
        axs_hausdorff[i].set_ylabel(f'Hausdorff Distance' if metric_name == 'HDRFDST' else f'Hausdorff distance 95th percentile')

        if metric_name in df.columns:
            data = [df[df['LABEL'] == label][metric_name] for label in df['LABEL'].unique()]
            axs_hausdorff[i].set_ylim(0, 90)  # Set y-axis limits for Hausdorff plots
            axs_hausdorff[i].boxplot(data, labels=custom_labels)
        else:
            axs_hausdorff[i].text(0.5, 0.5, f'Metric {metric_name} not available', ha='center', va='center', fontsize=12,
                                 color='red')

    plt.show()

if __name__ == '__main__':
    main()
