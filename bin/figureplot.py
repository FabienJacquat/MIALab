import matplotlib.pyplot as plt
import pandas as pd
import os


def main():

    directory_path = r'C:\Users\pierr\Desktop\Cours\BME3\Medical_Image_Analysis_Lab\MIALab_Fabien\bin\mia-result'

    # List of filenames
    filenames = ['result_summary.csv', 'result_summary_PP.csv']

    # Create a list for max_depth values
    max_depth = [5, 10, 20, 50, 75, 100, 110, 120, 140]

    for filename in filenames:

        # Dictionary to store data from each label for DICE
        dice_data_dict = {label: [] for label in ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

        # Dictionary to store data from each label for JACRD
        jacrd_data_dict = {label: [] for label in ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

        # Dictionary to store data from each label for HDRFDST
        hdrfdst_data_dict = {label: [] for label in ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

        # Dictionary to store data from each label for HDRFDST95
        hdrfdst95_data_dict = {label: [] for label in ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

        for root, dirs, files in os.walk(directory_path):
            for dir_name in dirs:
                # Construct the full path to the CSV file in the current directory
                file_path = os.path.join(root, dir_name, filename)

                # Check if the file exists
                if os.path.isfile(file_path):
                    # Read the CSV file using pandas
                    df = pd.read_csv(file_path, delimiter=';')

                    # Iterate over labels and filter the DataFrame for each label for DICE
                    for label in dice_data_dict.keys():
                        filtered_df_dice = df[
                            (df.iloc[:, 0] == label) & (df.iloc[:, 1] == 'DICE') & (df.iloc[:, 2] == 'MEAN')]
                        dice_data_dict[label].extend(filtered_df_dice.iloc[:, 3].values)

                    # Iterate over labels and filter the DataFrame for each label for JACRD
                    for label in jacrd_data_dict.keys():
                        filtered_df_jacrd = df[
                            (df.iloc[:, 0] == label) & (df.iloc[:, 1] == 'JACRD') & (df.iloc[:, 2] == 'MEAN')]
                        jacrd_data_dict[label].extend(filtered_df_jacrd.iloc[:, 3].values)

                    # Iterate over labels and filter the DataFrame for each label for HDRFDST
                    for label in hdrfdst_data_dict.keys():
                        filtered_df_hdrfdst = df[
                            (df.iloc[:, 0] == label) & (df.iloc[:, 1] == 'HDRFDST') & (df.iloc[:, 2] == 'MEAN')]
                        hdrfdst_data_dict[label].extend(filtered_df_hdrfdst.iloc[:, 3].values)

                    # Iterate over labels and filter the DataFrame for each label for HDRFDST95
                    for label in hdrfdst95_data_dict.keys():
                        filtered_df_hdrfdst95 = df[
                            (df.iloc[:, 0] == label) & (df.iloc[:, 1] == 'HDRFDST95') & (df.iloc[:, 2] == 'MEAN')]
                        if not filtered_df_hdrfdst95.empty:
                            hdrfdst95_data_dict[label].extend(filtered_df_hdrfdst95.iloc[:, 3].values)
                        else:
                            hdrfdst95_data_dict[label].append(0)  # Append 0 if no data is found

        # Plot the average data for max_depth
        # Plot the combined data for DICE and JACRD in a separate figure
        plt.figure(figsize=(12, 6))

        # Subplot for DICE
        plt.subplot(1, 2, 1)
        for label, data in dice_data_dict.items():
            if filename == 'result_summary.csv':
                plt.plot(max_depth[0:3], data[0:3], label=label)
            else:
                plt.plot(max_depth[0:3], data[0:3], label=label)

        plt.title(f'Average Dice coefficient per Max depth')
        plt.xlabel('Max depth')
        plt.ylabel('Average Dice coefficient')
        plt.legend()
        plt.ylim(0, 0.9)  # Set the y-axis upper limit

        # Subplot for JACRD
        plt.subplot(1, 2, 2)
        for label, data in jacrd_data_dict.items():
            if filename == 'result_summary.csv':
                plt.plot(max_depth[0:3], data[0:3], label=label)
            else:
                plt.plot(max_depth[0:3], data[0:3], label=label)

        plt.title(f'Average Jaccard coefficient per Max depth')
        plt.xlabel('Max depth')
        plt.ylabel('Average Jaccard coefficient')
        plt.legend()
        plt.ylim(0, 0.9)  # Set the y-axis upper limit

        # Add a title for the entire figure
        if filename == 'result_summary.csv':
            plt.suptitle('Result without post-processing')
        else:
            plt.suptitle('Result with post-processing')

        plt.tight_layout()

        # Plot the combined data for HDRFDST and HDRFDST95 in a separate figure
        plt.figure(figsize=(12, 6))

        # Subplot for HDRFDST
        plt.subplot(1, 2, 1)
        for label, data in hdrfdst_data_dict.items():
            if filename == 'result_summary.csv':
                plt.plot(max_depth[0:3], data[0:3], label=label)
            else:
                plt.plot(max_depth[0:3], data[0:3], label=label)

        plt.title(f'Average Hausdorff distance per Max depth')
        plt.xlabel('Max depth')
        plt.ylabel('Average Hausdorff distance')
        plt.legend()
        plt.ylim(0, 90)  # Set the y-axis upper limit

        # Subplot for HDRFDST95
        plt.subplot(1, 2, 2)
        for label, data in hdrfdst95_data_dict.items():
            if filename == 'result_summary.csv':
                plt.plot(max_depth[0:3], data[0:3], label=label)
            else:
                plt.plot(max_depth[0:3], data[0:3], label=label)

        plt.title(f'Average Hausdorff distance 95th per Max depth')
        plt.xlabel('Max depth')
        plt.ylabel('Average Hausdorff distance 95th percentile')
        plt.legend()
        plt.ylim(0, 90)  # Set the y-axis upper limit

        # Add a title for the entire figure
        if filename == 'result_summary.csv':
            plt.suptitle('Result without post-processing')
        else:
            plt.suptitle('Result with post-processing')

        plt.tight_layout()

        # ------------------------------------------------------------------------------------------------------
        # Plot the average data for nb_estimators

        # Create a list for nb_estimators
        nb_estimators = [50, 100, 150, 200, 300, 500, 600, 700, 800]

        # Plot the combined data for DICE and JACRD in a separate figure
        plt.figure(figsize=(12, 6))

        # Subplot for DICE
        plt.subplot(1, 2, 1)
        for label, data in dice_data_dict.items():
            if filename == 'result_summary.csv':
                plt.plot(nb_estimators[0:3], data[0:3], label=label)
            else:
                plt.plot(nb_estimators[0:3], data[0:3], label=label)

        plt.title(f'Average Dice coefficient per Numbers of estimators')
        plt.xlabel('Numbers of estimators')
        plt.ylabel('Average Dice coefficient')
        plt.legend()
        plt.ylim(0, 0.9)  # Set the y-axis upper limit

        # Subplot for JACRD
        plt.subplot(1, 2, 2)
        for label, data in jacrd_data_dict.items():
            if filename == 'result_summary.csv':
                plt.plot(nb_estimators[0:3], data[0:3], label=label)
            else:
                plt.plot(nb_estimators[0:3], data[0:3], label=label)

        plt.title(f'Average Jaccard coefficient per Numbers of estimators')
        plt.xlabel('Numbers of estimators')
        plt.ylabel('Average Jaccard coefficient')
        plt.legend()
        plt.ylim(0, 0.9)  # Set the y-axis upper limit

        # Add a title for the entire figure
        if filename == 'result_summary.csv':
            plt.suptitle('Result without post-processing')
        else:
            plt.suptitle('Result with post-processing')

        plt.tight_layout()

        # Plot the combined data for HDRFDST and HDRFDST95 in a separate figure
        plt.figure(figsize=(12, 6))

        # Subplot for HDRFDST
        plt.subplot(1, 2, 1)
        for label, data in hdrfdst_data_dict.items():
            if filename == 'result_summary.csv':
                plt.plot(nb_estimators[0:3], data[0:3], label=label)
            else:
                plt.plot(nb_estimators[0:3], data[0:3], label=label)

        plt.title(f'Average Hausdorff distance per Numbers of estimators')
        plt.xlabel('Numbers of estimators')
        plt.ylabel('Average Hausdorff distance')
        plt.legend()
        plt.ylim(0, 90)  # Set the y-axis upper limit

        # Subplot for HDRFDST95
        plt.subplot(1, 2, 2)
        for label, data in hdrfdst95_data_dict.items():
            if filename == 'result_summary.csv':
                plt.plot(nb_estimators[0:3], data[0:3], label=label)
            else:
                plt.plot(nb_estimators[0:3], data[0:3], label=label)


        plt.title(f'Average Hausdorff distance 95th per Numbers of estimators')
        plt.xlabel('Numbers of estimators')
        plt.ylabel('Average Hausdorff distance 95th percentile')
        plt.legend()
        plt.ylim(0, 90)  # Set the y-axis upper limit

        # Add a title for the entire figure
        if filename == 'result_summary.csv':
            plt.suptitle('Result without post-processing')
        else:
            plt.suptitle('Result with post-processing')

        plt.tight_layout()

        # ------------------------------------------------------------------------------------------------------
        # Plot the average data for class weights

        # Create a list for class weights
        class_weights = ['No class weight', '{1, 1, 1.5, 1.5, 1.5}', '{1, 1.01, 1.16, 1.5, 1.16}']

        # Plot the combined data for DICE and JACRD in a separate figure
        plt.figure(figsize=(12, 6))

        # Subplot for DICE
        plt.subplot(1, 2, 1)
        for label, data in dice_data_dict.items():
            plt.plot(class_weights, data[0:3], label=label)  # Adjust the range [4:7] based on your data length

        plt.title(f'Average Dice coefficient per Class Weights')
        plt.xlabel('Class Weights')
        plt.gca().xaxis.labelpad = 10
        plt.ylabel('Average Dice coefficient')
        plt.legend()
        plt.ylim(0, 0.9)  # Set the y-axis upper limit

        # Subplot for JACRD
        plt.subplot(1, 2, 2)
        for label, data in jacrd_data_dict.items():
            if filename == 'result_summary.csv':
                plt.plot(class_weights, data[0:3], label=label)
            else:
                plt.plot(class_weights, data[0:3], label=label)

        plt.title(f'Average Jaccard coefficient per Class Weights')
        plt.xlabel('Class Weights')
        plt.gca().xaxis.labelpad = 10
        plt.ylabel('Average Jaccard coefficient')
        plt.legend()
        plt.ylim(0, 0.9)  # Set the y-axis upper limit

        # Add a title for the entire figure
        if filename == 'result_summary.csv':
            plt.suptitle('Result without post-processing')
        else:
            plt.suptitle('Result with post-processing')

        plt.tight_layout()

        # Plot the combined data for HDRFDST and HDRFDST95 in a separate figure
        plt.figure(figsize=(12, 6))

        # Subplot for HDRFDST
        plt.subplot(1, 2, 1)
        for label, data in hdrfdst_data_dict.items():
            if filename == 'result_summary.csv':
                plt.plot(class_weights, data[0:3], label=label)
            else:
                plt.plot(class_weights, data[0:3], label=label)

        plt.title(f'Average Hausdorff distance per Class Weights')
        plt.xlabel('Class Weights')
        plt.gca().xaxis.labelpad = 10
        plt.ylabel('Average Hausdorff distance')
        plt.legend()
        plt.ylim(0, 90)  # Set the y-axis upper limit

        # Subplot for HDRFDST95
        plt.subplot(1, 2, 2)
        for label, data in hdrfdst95_data_dict.items():
            # Add a title for the entire figure
            if filename == 'result_summary.csv':
                plt.plot(class_weights, data[0:3], label=label)
            else:
                plt.plot(class_weights, data[0:3], label=label)


        plt.title(f'Average Hausdorff distance 95th per Class Weights')
        plt.xlabel('Class Weights')
        plt.gca().xaxis.labelpad = 10
        plt.ylabel('Average Hausdorff distance 95th percentile')
        plt.legend()
        plt.ylim(0, 90)  # Set the y-axis upper limit

        # Add a title for the entire figure
        if filename == 'result_summary.csv':
            plt.suptitle('Result without post-processing')
        else:
            plt.suptitle('Result with post-processing')

        plt.tight_layout()

        plt.show()

if __name__ == '__main__':
    main()
