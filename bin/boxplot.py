import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def main():

    directory_path = r'..\bin\mia-result'

    # Get the file paths of two tests you want to compare
    result_name1 = '0-2023-12-14-16-28-41'
    file_path1 = os.path.join(directory_path, result_name1, 'results.csv')
    result_name2 = '1-2023-12-14-16-03-39'
    file_path2 = os.path.join(directory_path, result_name2, 'results.csv')

    # Filter dataset 1 by processing groups
    df1 = pd.read_csv(file_path1, delimiter=';')
    df1_notPP = df1[~df1['SUBJECT'].str.contains('PP')]
    df1_PP = df1[df1['SUBJECT'].str.contains('PP')]

    # Filter dataset 2 by processing groups
    df2 = pd.read_csv(file_path2, delimiter=';')
    df2_notPP = df2[~df2['SUBJECT'].str.contains('PP')]
    df2_PP = df2[df2['SUBJECT'].str.contains('PP')]

    # Choose which datasets you want to compare
    data1 = df1_notPP
    data2 = df2_notPP

    # List which metrics you want to compare
    dice_jaccard_metrics = ['DICE', 'JACRD']
    hausdorff_metrics = ['HDRFDST', 'HDRFDST95']

    # Create separate figures for Dice/Jaccard and Hausdorff metrics
    fig_dice_jaccard, axs_dice_jaccard = plt.subplots(1, 2, figsize=(12, 6))
    fig_hausdorff, axs_hausdorff = plt.subplots(1, 2, figsize=(12, 6))

    # Adjust the space between subplots
    plt.subplots_adjust(top=0.88, hspace=0.3)  # Increased vertical space

    # Custom x-axis labels
    custom_labels = ['Amygdala', 'Grey', 'Hippo', 'Thalamus', 'White']

    # Fix formatting for boxplots
    positions = np.array(range(len(custom_labels)))
    width = 0.4  # Adjust the width as needed
    color_data1 = 'skyblue'
    color_data2 = 'mediumorchid'

    # Plot Dice and Jaccard coefficients
    for i, metric_name in enumerate(dice_jaccard_metrics, 0):
        axs_dice_jaccard[i].set_title(f'Dice per Label' if metric_name == 'DICE' else f'Jaccard per Label')
        axs_dice_jaccard[i].set_xlabel('')  # Set x-axis label to an empty string
        axs_dice_jaccard[i].set_ylabel(f'Dice coefficient' if metric_name == 'DICE' else f'Jaccard coefficient')

        if metric_name in data1.columns:
            filtered_data1 = [data1[data1['LABEL'] == label][metric_name] for label in data1['LABEL'].unique()]
            filtered_data2 = [data2[data2['LABEL'] == label][metric_name] for label in data2['LABEL'].unique()]
            axs_dice_jaccard[i].set_ylim(0, 1)  # Set y-axis limits for Dice and Jaccard plots
            axs_dice_jaccard[i].boxplot(filtered_data1, positions=positions - width / 2, widths=width, patch_artist=True, boxprops=dict(facecolor=color_data1))
            axs_dice_jaccard[i].boxplot(filtered_data2, positions=positions + width / 2, widths=width, patch_artist=True, boxprops=dict(facecolor=color_data2))
        else:
            axs_dice_jaccard[i].text(0.5, 0.5, f'Metric {metric_name} not available', ha='center', va='center', fontsize=12,
                                   color='red')

        axs_dice_jaccard[i].set_xticks(positions)
        axs_dice_jaccard[i].set_xticklabels(custom_labels)

    # Plot Hausdorff and 95th Hausdorff coefficients
    for i, metric_name in enumerate(hausdorff_metrics, 0):
        axs_hausdorff[i].set_title(f'Hausdorff Distance per Label' if metric_name == 'HDRFDST' else f'Hausdorff distance 95th per Label')
        axs_hausdorff[i].set_xlabel('')  # Set x-axis label to an empty string
        axs_hausdorff[i].set_ylabel(f'Hausdorff Distance' if metric_name == 'HDRFDST' else f'Hausdorff distance 95th percentile')

        if metric_name in data1.columns:
            filtered_data1 = [data1[data1['LABEL'] == label][metric_name] for label in data1['LABEL'].unique()]
            filtered_data2 = [data2[data2['LABEL'] == label][metric_name] for label in data2['LABEL'].unique()]
            axs_hausdorff[i].set_ylim(0, 90)  # Set y-axis limits for Hausdorff plots
            axs_hausdorff[i].boxplot(filtered_data1, positions=positions - width / 2, widths=width, patch_artist=True, boxprops=dict(facecolor=color_data1))
            axs_hausdorff[i].boxplot(filtered_data2, positions=positions + width / 2, widths=width, patch_artist=True, boxprops=dict(facecolor=color_data2))
        else:
            axs_hausdorff[i].text(0.5, 0.5, f'Metric {metric_name} not available', ha='center', va='center', fontsize=12,
                                 color='red')

        axs_hausdorff[i].set_xticks(positions)
        axs_hausdorff[i].set_xticklabels(custom_labels)

    # Create legend elements for each figure
    legend_elements_dice = [plt.Rectangle((0, 0), 1, 1, color=color_data1, label='data1'),
                            plt.Rectangle((0, 0), 1, 1, color=color_data2, label='data2')]
    legend_elements_hausdorff = [plt.Rectangle((0, 0), 1, 1, color=color_data1, label='data1'),
                                 plt.Rectangle((0, 0), 1, 1, color=color_data2, label='data2')]

    # Show legend for Dice/Jaccard figure
    fig_dice_jaccard.legend(handles=legend_elements_dice, loc='upper left')

    # Show legend for Hausdorff figure
    fig_hausdorff.legend(handles=legend_elements_hausdorff, loc='upper left')

    plt.show()

if __name__ == '__main__':
    main()