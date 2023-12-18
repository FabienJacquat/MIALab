import matplotlib.pyplot as plt
import pandas as pd
import os


def main():

    directory_path = r'C:\Users\pierr\Desktop\Cours\BME3\Medical_Image_Analysis_Lab\MIALab_Fabien\bin\mia-result'

    # List of filenames
    filenames = ['result_summary.csv', 'result_summary_PP.csv']

    # List of classifier
    classifiers = ['forest', 'extremely']

    # Create a list for max_depth values
    max_depth = [5, 10, 25, 50, 75, 100]

    for classifier in classifiers:
        for filename in filenames:

            # Dictionary to store data from each label for DICE
            dice_data_dict = {label: [] for label in ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            # Dictionary to store data from each label for JACRD
            jacrd_data_dict = {label: [] for label in ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            # Dictionary to store data from each label for HDRFDST
            hdrfdst_data_dict = {label: [] for label in ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            # Dictionary to store data from each label for HDRFDST95
            hdrfdst95_data_dict = {label: [] for label in ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}


            idx_dice = {label: [] for label in
                                   ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_jacrd = {label: [] for label in
                                   ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_hdrfdst = {label: [] for label in
                                   ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_hdrfdst95 = {label: [] for label in
                                   ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_dice_max_depth = {label: [] for label in
                        ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_jacrd_max_depth = {label: [] for label in
                         ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_hdrfdst_max_depth = {label: [] for label in
                           ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_hdrfdst95_max_depth = {label: [] for label in
                             ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_dice_nb_estimators = {label: [] for label in
                        ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_jacrd_nb_estimators = {label: [] for label in
                         ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_hdrfdst_nb_estimators = {label: [] for label in
                           ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}

            idx_hdrfdst95_nb_estimators = {label: [] for label in
                             ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']}



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

                            # Add the index of dir_name to idx_dice
                            for value in filtered_df_dice.iloc[:, 3].values:
                                if value == 0 or value == float('inf'):
                                    idx_dice[label].append(dirs.index(dir_name))


                        # Iterate over labels and filter the DataFrame for each label for JACRD
                        for label in jacrd_data_dict.keys():
                            filtered_df_jacrd = df[
                                (df.iloc[:, 0] == label) & (df.iloc[:, 1] == 'JACRD') & (df.iloc[:, 2] == 'MEAN')]
                            jacrd_data_dict[label].extend(filtered_df_jacrd.iloc[:, 3].values)

                            # Add the index of dir_name to idx_dice
                            for value in filtered_df_jacrd.iloc[:, 3].values:
                                if value == 0 or value == float('inf'):
                                    idx_jacrd[label].append(dirs.index(dir_name))


                                    # Iterate over labels and filter the DataFrame for each label for HDRFDST
                        for label in hdrfdst_data_dict.keys():
                            filtered_df_hdrfdst = df[
                                (df.iloc[:, 0] == label) & (df.iloc[:, 1] == 'HDRFDST') & (df.iloc[:, 2] == 'MEAN')]
                            hdrfdst_data_dict[label].extend(filtered_df_hdrfdst.iloc[:, 3].values)

                            # Add the index of dir_name to idx_dice
                            for value in filtered_df_hdrfdst.iloc[:, 3].values:
                                if value == 0 or value == float('inf'):
                                    idx_hdrfdst[label].append(dirs.index(dir_name))

                        # Iterate over labels and filter the DataFrame for each label for HDRFDST95
                        for label in hdrfdst95_data_dict.keys():
                            filtered_df_hdrfdst95 = df[
                                (df.iloc[:, 0] == label) & (df.iloc[:, 1] == 'HDRFDST95') & (df.iloc[:, 2] == 'MEAN')]
                            hdrfdst95_data_dict[label].extend(filtered_df_hdrfdst95.iloc[:, 3].values)

                            # Add the index of dir_name to idx_dice
                            for value in filtered_df_hdrfdst95.iloc[:, 3].values:
                                if value == 0 or value == float('inf'):
                                    idx_hdrfdst95[label].append(dirs.index(dir_name))

            # Plot the average data for max_depth
            # Plot the combined data for DICE and JACRD in a separate figure
            plt.figure(figsize=(12, 6))

            # Subplot for DICE
            plt.subplot(1, 2, 1)
            for label, data in dice_data_dict.items():
                if classifier == 'forest':

                    # Filter data based on idx_dice[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_dice[label] or (index in idx_dice[label] and not (19 <= index < 25))]

                    # Subtract 19 from every element in idx_dice[label]
                    idx_dice_max_depth[label] = [index - 19 for index in idx_dice[label]]

                    # Remove negative values from idx_dice_max_depth[label]
                    idx_dice_max_depth[label] = [index for index in idx_dice_max_depth[label] if index >= 0]

                    # Filter max_depth based on idx_dice_max_depth[label]
                    filtered_max_depth = [value for index, value in enumerate(max_depth) if
                                          index not in idx_dice_max_depth[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 25 - len([x for x in idx_dice[label] if isinstance(x, int) and 19 <= x < 25])

                    # Plot the filtered data and filtered_max_depth
                    plt.plot(filtered_max_depth, filtered_data[19:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)
                else:
                    # Filter data based on idx_dice[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_dice[label] or (index in idx_dice[label] and not (25 <= index < 31))]

                    # Subtract 19 from every element in idx_dice[label]
                    idx_dice_max_depth[label] = [index - 25 for index in idx_dice[label]]

                    # Remove negative values from idx_dice_max_depth[label]
                    idx_dice_max_depth[label] = [index for index in idx_dice_max_depth[label] if index >= 0]

                    # Filter max_depth based on idx_dice_max_depth[label]
                    filtered_max_depth = [value for index, value in enumerate(max_depth) if
                                          index not in idx_dice_max_depth[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 31 - len([x for x in idx_dice[label] if isinstance(x, int) and 25 <= x < 31])

                    # Plot the filtered data and filtered_max_depth
                    plt.plot(filtered_max_depth, filtered_data[25:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)

            plt.title(f'Average Dice coefficient per Max depth')
            plt.xlabel('Max depth')
            plt.ylabel('Average Dice coefficient')
            plt.legend()
            plt.ylim(0, 0.9)  # Set the y-axis upper limit

            # Subplot for JACRD
            plt.subplot(1, 2, 2)
            for label, data in jacrd_data_dict.items():
                if classifier == 'forest':

                    # Filter data based on idx_jacrd[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_jacrd[label] or (index in idx_jacrd[label] and not (19 <= index < 25))]

                    # Subtract 19 from every element in idx_jacrd[label]
                    idx_jacrd_max_depth[label] = [index - 19 for index in idx_jacrd[label]]

                    # Remove negative values from idx_jacrd_max_depth[label]
                    idx_jacrd_max_depth[label] = [index for index in idx_jacrd_max_depth[label] if index >= 0]

                    # Filter max_depth based on idx_jacrd_max_depth[label]
                    filtered_max_depth = [value for index, value in enumerate(max_depth) if
                                          index not in idx_jacrd_max_depth[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 25 - len([x for x in idx_jacrd[label] if isinstance(x, int) and 19 <= x < 25])

                    # Plot the filtered data and filtered_max_depth
                    plt.plot(filtered_max_depth, filtered_data[19:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)
                else:
                    # Filter data based on idx_jacrd[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_jacrd[label] or (index in idx_jacrd[label] and not (25 <= index < 31))]

                    # Subtract 19 from every element in idx_jacrd[label]
                    idx_jacrd_max_depth[label] = [index - 25 for index in idx_jacrd[label]]

                    # Remove negative values from idx_jacrd_max_depth[label]
                    idx_jacrd_max_depth[label] = [index for index in idx_jacrd_max_depth[label] if index >= 0]

                    # Filter max_depth based on idx_jacrd_max_depth[label]
                    filtered_max_depth = [value for index, value in enumerate(max_depth) if
                                          index not in idx_jacrd_max_depth[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 31 - len([x for x in idx_jacrd[label] if isinstance(x, int) and 25 <= x < 31])

                    # Plot the filtered data and filtered_max_depth
                    plt.plot(filtered_max_depth, filtered_data[25:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)

            plt.title(f'Average Jaccard coefficient per Max depth')
            plt.xlabel('Max depth')
            plt.ylabel('Average Jaccard coefficient')
            plt.legend()
            plt.ylim(0, 0.9)  # Set the y-axis upper limit


            # Add a title for the entire figure
            if filename == 'result_summary.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: No\n classifier: random forest')
            elif filename == 'result_summary_PP.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: Yes\n classifier: random forest')
            elif filename == 'result_summary.csv' and classifier == 'extremely':
                plt.suptitle('Post-processing: No\n classifier: extremely randomized trees')
            else:
                plt.suptitle('Post-processing: Yes\n classifier: extremely randomized trees')

            plt.tight_layout()

            # Plot the combined data for HDRFDST and HDRFDST95 in a separate figure
            plt.figure(figsize=(12, 6))

            # Subplot for HDRFDST
            plt.subplot(1, 2, 1)
            for label, data in hdrfdst_data_dict.items():
                if classifier == 'forest':

                    # Filter data based on idx_hdrfdst[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_hdrfdst[label] or (index in idx_hdrfdst[label] and not (19 <= index < 25))]

                    # Subtract 19 from every element in idx_hdrfdst[label]
                    idx_hdrfdst_max_depth[label] = [index - 19 for index in idx_hdrfdst[label]]

                    # Remove negative values from idx_hdrfdst_max_depth[label]
                    idx_hdrfdst_max_depth[label] = [index for index in idx_hdrfdst_max_depth[label] if index >= 0]

                    # Filter max_depth based on idx_hdrfdst_max_depth[label]
                    filtered_max_depth = [value for index, value in enumerate(max_depth) if
                                          index not in idx_hdrfdst_max_depth[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 25 - len([x for x in idx_hdrfdst[label] if isinstance(x, int) and 19 <= x < 25])

                    # Plot the filtered data and filtered_max_depth
                    plt.plot(filtered_max_depth, filtered_data[19:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)
                else:
                    # Filter data based on idx_hdrfdst[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_hdrfdst[label] or (index in idx_hdrfdst[label] and not (25 <= index < 31))]

                    # Subtract 19 from every element in idx_hdrfdst[label]
                    idx_hdrfdst_max_depth[label] = [index - 25 for index in idx_hdrfdst[label]]

                    # Remove negative values from idx_hdrfdst_max_depth[label]
                    idx_dice_max_depth[label] = [index for index in idx_hdrfdst_max_depth[label] if index >= 0]

                    # Filter max_depth based on idx_hdrfdst_max_depth[label]
                    filtered_max_depth = [value for index, value in enumerate(max_depth) if
                                          index not in idx_hdrfdst_max_depth[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 31 - len([x for x in idx_hdrfdst[label] if isinstance(x, int) and 25 <= x < 31])

                    # Plot the filtered data and filtered_max_depth
                    plt.plot(filtered_max_depth, filtered_data[25:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)

            plt.title(f'Average Hausdorff distance per Max depth')
            plt.xlabel('Max depth')
            plt.ylabel('Average Hausdorff distance')
            plt.legend()
            plt.ylim(0, 90)  # Set the y-axis upper limit

            # Subplot for HDRFDST95
            plt.subplot(1, 2, 2)
            for label, data in hdrfdst95_data_dict.items():
                if classifier == 'forest':

                    # Filter data based on idx_hdrfdst95[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_hdrfdst95[label] or (index in idx_hdrfdst95[label] and not (19 <= index < 25))]

                    # Subtract 19 from every element in idx_hdrfdst95[label]
                    idx_hdrfdst95_max_depth[label] = [index - 19 for index in idx_hdrfdst95[label]]

                    # Remove negative values from idx_hdrfdst95_max_depth[label]
                    idx_hdrfdst95_max_depth[label] = [index for index in idx_hdrfdst95_max_depth[label] if index >= 0]

                    # Filter max_depth based on idx_hdrfdst95_max_depth[label]
                    filtered_max_depth = [value for index, value in enumerate(max_depth) if
                                          index not in idx_hdrfdst95_max_depth[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 25 - len([x for x in idx_hdrfdst95[label] if isinstance(x, int) and 19 <= x < 25])

                    # Plot the filtered data and filtered_max_depth
                    plt.plot(filtered_max_depth, filtered_data[19:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)
                else:
                    # Filter data based on idx_hdrfdst95[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_hdrfdst95[label] or (index in idx_hdrfdst95[label] and not (25 <= index < 31))]

                    # Subtract 19 from every element in idx_hdrfdst95[label]
                    idx_hdrfdst95_max_depth[label] = [index - 25 for index in idx_hdrfdst95[label]]

                    # Remove negative values from idx_hdrfdst95_max_depth[label]
                    idx_hdrfdst95_max_depth[label] = [index for index in idx_hdrfdst95_max_depth[label] if index >= 0]

                    # Filter max_depth based on idx_hdrfdst95_max_depth[label]
                    filtered_max_depth = [value for index, value in enumerate(max_depth) if
                                          index not in idx_hdrfdst95_max_depth[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 31 - len([x for x in idx_hdrfdst95[label] if isinstance(x, int) and 25 <= x < 31])

                    # Plot the filtered data and filtered_max_depth
                    plt.plot(filtered_max_depth, filtered_data[25:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)

            plt.title(f'Average Hausdorff distance 95th per Max depth')
            plt.xlabel('Max depth')
            plt.ylabel('Average Hausdorff distance 95th percentile')
            plt.legend()
            plt.ylim(0, 90)  # Set the y-axis upper limit

            # Add a title for the entire figure
            if filename == 'result_summary.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: No\n classifier: random forest')
            elif filename == 'result_summary_PP.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: Yes\n classifier: random forest')
            elif filename == 'result_summary.csv' and classifier == 'extremely':
                plt.suptitle('Post-processing: No\n classifier: extremely randomized trees')
            else:
                plt.suptitle('Post-processing: Yes\n classifier: extremely randomized trees')

            plt.tight_layout()

            # ------------------------------------------------------------------------------------------------------
            # Plot the average data for nb_estimators

            # Create a list for nb_estimators
            nb_estimators = [5, 20, 50, 100, 150, 300]

            # Plot the combined data for DICE and JACRD in a separate figure
            plt.figure(figsize=(12, 6))

            # Subplot for DICE
            plt.subplot(1, 2, 1)
            for label, data in dice_data_dict.items():
                if classifier == 'forest':

                    # Filter data based on idx_dice[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_dice[label] or (index in idx_dice[label] and not (7 <= index < 13))]

                    # Subtract 19 from every element in idx_dice[label]
                    idx_dice_nb_estimators[label] = [index - 7 for index in idx_dice[label]]

                    # Remove negative values from idx_dice_nb_estimators[label]
                    idx_dice_nb_estimators[label] = [index for index in idx_dice_nb_estimators[label] if index >= 0]

                    # Filter nb_estimators based on idx_dice_nb_estimators[label]
                    filtered_nb_estimators = [value for index, value in enumerate(nb_estimators) if
                                          index not in idx_dice_nb_estimators[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 13 - len([x for x in idx_dice[label] if isinstance(x, int) and 7 <= x < 13])

                    # Plot the filtered data and filtered_nb_estimators
                    plt.plot(filtered_nb_estimators, filtered_data[7:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)
                else:
                    # Filter data based on idx_dice[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_dice[label] or (index in idx_dice[label] and not (13 <= index < 19))]

                    # Subtract 19 from every element in idx_dice[label]
                    idx_dice_nb_estimators[label] = [index - 13 for index in idx_dice[label]]

                    # Remove negative values from idx_dice_nb_estimators[label]
                    idx_dice_nb_estimators[label] = [index for index in idx_dice_nb_estimators[label] if index >= 0]

                    # Filter nb_estimators based on idx_dice_nb_estimators[label]
                    filtered_nb_estimators = [value for index, value in enumerate(nb_estimators) if
                                          index not in idx_dice_nb_estimators[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 19 - len([x for x in idx_dice[label] if isinstance(x, int) and 13 <= x < 19])

                    # Plot the filtered data and filtered_nb_estimators
                    plt.plot(filtered_nb_estimators, filtered_data[13:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)

            plt.title(f'Average Dice coefficient per Numbers of estimators')
            plt.xlabel('Numbers of estimators')
            plt.ylabel('Average Dice coefficient')
            plt.legend()
            plt.ylim(0, 0.9)  # Set the y-axis upper limit

            # Subplot for JACRD
            plt.subplot(1, 2, 2)
            for label, data in jacrd_data_dict.items():
                if classifier == 'forest':

                    # Filter data based on idx_jacrd[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_jacrd[label] or (index in idx_jacrd[label] and not (7 <= index < 13))]

                    # Subtract 19 from every element in idx_jacrd[label]
                    idx_jacrd_nb_estimators[label] = [index - 7 for index in idx_jacrd[label]]

                    # Remove negative values from idx_jacrd_nb_estimators[label]
                    idx_jacrd_nb_estimators[label] = [index for index in idx_jacrd_nb_estimators[label] if index >= 0]

                    # Filter nb_estimators based on idx_jacrd_nb_estimators[label]
                    filtered_nb_estimators = [value for index, value in enumerate(nb_estimators) if
                                          index not in idx_jacrd_nb_estimators[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 13 - len([x for x in idx_jacrd[label] if isinstance(x, int) and 7 <= x < 13])

                    # Plot the filtered data and filtered_nb_estimators
                    plt.plot(filtered_nb_estimators, filtered_data[7:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)
                else:
                    # Filter data based on idx_jacrd[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_jacrd[label] or (index in idx_jacrd[label] and not (13 <= index < 19))]

                    # Subtract 19 from every element in idx_jacrd[label]
                    idx_jacrd_nb_estimators[label] = [index - 13 for index in idx_jacrd[label]]

                    # Remove negative values from idx_jacrd_nb_estimators[label]
                    idx_jacrd_nb_estimators[label] = [index for index in idx_jacrd_nb_estimators[label] if index >= 0]

                    # Filter nb_estimators based on idx_jacrd_nb_estimators[label]
                    filtered_nb_estimators = [value for index, value in enumerate(nb_estimators) if
                                          index not in idx_jacrd_nb_estimators[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 19 - len([x for x in idx_jacrd[label] if isinstance(x, int) and 13 <= x < 19])

                    # Plot the filtered data and filtered_nb_estimators
                    plt.plot(filtered_nb_estimators, filtered_data[13:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)

            plt.title(f'Average Jaccard coefficient per Numbers of estimators')
            plt.xlabel('Numbers of estimators')
            plt.ylabel('Average Jaccard coefficient')
            plt.legend()
            plt.ylim(0, 0.9)  # Set the y-axis upper limit

            # Add a title for the entire figure
            if filename == 'result_summary.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: No\n classifier: random forest')
            elif filename == 'result_summary_PP.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: Yes\n classifier: random forest')
            elif filename == 'result_summary.csv' and classifier == 'extremely':
                plt.suptitle('Post-processing: No\n classifier: extremely randomized trees')
            else:
                plt.suptitle('Post-processing: Yes\n classifier: extremely randomized trees')

            plt.tight_layout()

            # Plot the combined data for HDRFDST and HDRFDST95 in a separate figure
            plt.figure(figsize=(12, 6))

            # Subplot for HDRFDST
            plt.subplot(1, 2, 1)
            for label, data in hdrfdst_data_dict.items():
                if classifier == 'forest':

                    # Filter data based on idx_hdrfdst[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_hdrfdst[label] or (index in idx_hdrfdst[label] and not (7 <= index < 13))]

                    # Subtract 19 from every element in idx_hdrfdst[label]
                    idx_hdrfdst_nb_estimators[label] = [index - 7 for index in idx_hdrfdst[label]]

                    # Remove negative values from idx_hdrfdst_nb_estimators[label]
                    idx_hdrfdst_nb_estimators[label] = [index for index in idx_hdrfdst_nb_estimators[label] if index >= 0]

                    # Filter nb_estimators based on idx_hdrfdst_nb_estimators[label]
                    filtered_nb_estimators = [value for index, value in enumerate(nb_estimators) if
                                          index not in idx_hdrfdst_nb_estimators[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 13 - len([x for x in idx_hdrfdst[label] if isinstance(x, int) and 7 <= x < 13])

                    # Plot the filtered data and filtered_nb_estimators
                    plt.plot(filtered_nb_estimators, filtered_data[7:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)
                else:
                    # Filter data based on idx_hdrfdst[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_hdrfdst[label] or (index in idx_hdrfdst[label] and not (13 <= index < 19))]

                    # Subtract 19 from every element in idx_hdrfdst[label]
                    idx_hdrfdst_nb_estimators[label] = [index - 13 for index in idx_hdrfdst[label]]

                    # Remove negative values from idx_hdrfdst_nb_estimators[label]
                    idx_dice_nb_estimators[label] = [index for index in idx_hdrfdst_nb_estimators[label] if index >= 0]

                    # Filter max_depth based on idx_hdrfdst_nb_estimators[label]
                    filtered_nb_estimators = [value for index, value in enumerate(nb_estimators) if
                                          index not in idx_hdrfdst_nb_estimators[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 19 - len([x for x in idx_hdrfdst[label] if isinstance(x, int) and 13 <= x < 19])

                    # Plot the filtered data and filtered_nb_estimators
                    plt.plot(filtered_nb_estimators, filtered_data[13:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)

            plt.title(f'Average Hausdorff distance per Numbers of estimators')
            plt.xlabel('Numbers of estimators')
            plt.ylabel('Average Hausdorff distance')
            plt.legend()
            plt.ylim(0, 90)  # Set the y-axis upper limit

            # Subplot for HDRFDST95
            plt.subplot(1, 2, 2)
            for label, data in hdrfdst95_data_dict.items():
                if classifier == 'forest':

                    # Filter data based on idx_hdrfdst95[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_hdrfdst95[label] or (index in idx_hdrfdst95[label] and not (7 <= index < 13))]

                    # Subtract 19 from every element in idx_hdrfdst95[label]
                    idx_hdrfdst95_nb_estimators[label] = [index - 7 for index in idx_hdrfdst95[label]]

                    # Remove negative values from idx_hdrfdst95_nb_estimators[label]
                    idx_hdrfdst95_nb_estimators[label] = [index for index in idx_hdrfdst95_nb_estimators[label] if index > 0]

                    # Filter nb_estimators based on idx_hdrfdst95_nb_estimators[label]
                    filtered_nb_estimators = [value for index, value in enumerate(nb_estimators) if
                                          index not in idx_hdrfdst95_nb_estimators[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 13 - len([x for x in idx_hdrfdst95[label] if isinstance(x, int) and 7 <= x < 13])

                    # Plot the filtered data and filtered_nb_estimators
                    plt.plot(filtered_nb_estimators, filtered_data[7:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)
                else:
                    # Filter data based on idx_hdrfdst95[label]
                    filtered_data = [value for index, value in enumerate(data) if
                                     index not in idx_hdrfdst95[label] or (index in idx_hdrfdst95[label] and not (13 <= index < 19))]

                    # Subtract 19 from every element in idx_hdrfdst95[label]
                    idx_hdrfdst95_nb_estimators[label] = [index - 13 for index in idx_hdrfdst95[label]]

                    # Remove negative values from idx_hdrfdst95_nb_estimators[label]
                    idx_hdrfdst95_nb_estimators[label] = [index for index in idx_hdrfdst95_nb_estimators[label] if index >= 0]

                    # Filter nb_estimators based on idx_hdrfdst95_nb_estimators[label]
                    filtered_nb_estimators = [value for index, value in enumerate(nb_estimators) if
                                          index not in idx_hdrfdst95_nb_estimators[label]]

                    # Create end_data by filtering elements between 0 and 25, and calculate its length
                    end_data = 19 - len([x for x in idx_hdrfdst95[label] if isinstance(x, int) and 13 <= x < 19])

                    # Plot the filtered data and filtered_nb_estimators
                    plt.plot(filtered_nb_estimators, filtered_data[13:end_data], label=label, marker='x', markersize=6,
                             linestyle='-', linewidth=1)


            plt.title(f'Average Hausdorff distance 95th per Numbers of estimators')
            plt.xlabel('Numbers of estimators')
            plt.ylabel('Average Hausdorff distance 95th percentile')
            plt.legend()
            plt.ylim(0, 90)  # Set the y-axis upper limit

            # Add a title for the entire figure
            if filename == 'result_summary.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: No\n classifier: random forest')
            elif filename == 'result_summary_PP.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: Yes\n classifier: random forest')
            elif filename == 'result_summary.csv' and classifier == 'extremely':
                plt.suptitle('Post-processing: No\n classifier: extremely randomized trees')
            else:
                plt.suptitle('Post-processing: Yes\n classifier: extremely randomized trees')

            plt.tight_layout()

            # ------------------------------------------------------------------------------------------------------
            # Plot the average data for class weights

            # Create a list for class weights
            class_weights = ['No class weight', '{1, 1, 1.5, 1.5, 1.5}', '{1, 1.01, 1.16, 1.5, 1.16}',
                             '{1, 1.25, 5.54, 15.12, 5.46}']

            # Plot the combined data for DICE and JACRD in a separate figure
            plt.figure(figsize=(12, 6))

            # Subplot for DICE
            plt.subplot(1, 2, 1)
            for label, data in dice_data_dict.items():
                plt.plot(class_weights, data[0:4], label=label, marker='x', markersize=6, linestyle='-', linewidth=1)  # Adjust the range [4:7] based on your data length

            plt.title(f'Average Dice coefficient per Class Weights')
            plt.xlabel('Class Weights')
            plt.gca().xaxis.labelpad = 10
            plt.ylabel('Average Dice coefficient')
            plt.legend()
            plt.ylim(0, 0.9)  # Set the y-axis upper limit
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels diagonally

            # Subplot for JACRD
            plt.subplot(1, 2, 2)
            for label, data in jacrd_data_dict.items():
                if classifier == 'forest':
                    plt.plot(class_weights, data[0:4], label=label, marker='x', markersize=6, linestyle='-', linewidth=1)
                else:
                    plt.plot(class_weights, data[4:8], label=label, marker='x', markersize=6, linestyle='-', linewidth=1)

            plt.title(f'Average Jaccard coefficient per Class Weights')
            plt.xlabel('Class Weights')
            plt.gca().xaxis.labelpad = 10
            plt.ylabel('Average Jaccard coefficient')
            plt.legend()
            plt.ylim(0, 0.9)  # Set the y-axis upper limit
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels diagonally

            # Add a title for the entire figure
            if filename == 'result_summary.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: No\n classifier: random forest')
            elif filename == 'result_summary_PP.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: Yes\n classifier: random forest')
            elif filename == 'result_summary.csv' and classifier == 'extremely':
                plt.suptitle('Post-processing: No\n classifier: extremely randomized trees')
            else:
                plt.suptitle('Post-processing: Yes\n classifier: extremely randomized trees')

            plt.tight_layout()

            # Plot the combined data for HDRFDST and HDRFDST95 in a separate figure
            plt.figure(figsize=(12, 6))

            # Subplot for HDRFDST
            plt.subplot(1, 2, 1)
            for label, data in hdrfdst_data_dict.items():
                if classifier == 'forest':
                    plt.plot(class_weights, data[0:4], label=label, marker='x', markersize=6, linestyle='-', linewidth=1)
                else:
                    plt.plot(class_weights, data[4:8], label=label, marker='x', markersize=6, linestyle='-', linewidth=1)

            plt.title(f'Average Hausdorff distance per Class Weights')
            plt.xlabel('Class Weights')
            plt.gca().xaxis.labelpad = 10
            plt.ylabel('Average Hausdorff distance')
            plt.legend()
            plt.ylim(0, 90)  # Set the y-axis upper limit
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels diagonally

            # Subplot for HDRFDST95
            plt.subplot(1, 2, 2)
            for label, data in hdrfdst95_data_dict.items():
                # Add a title for the entire figure
                if classifier == 'forest':
                    plt.plot(class_weights, data[0:4], label=label, marker='x', markersize=6, linestyle='-', linewidth=1)
                else:
                    plt.plot(class_weights, data[4:8], label=label, marker='x', markersize=6, linestyle='-', linewidth=1)

            plt.title(f'Average Hausdorff distance 95th per Class Weights')
            plt.xlabel('Class Weights')
            plt.gca().xaxis.labelpad = 10
            plt.ylabel('Average Hausdorff distance 95th percentile')
            plt.legend()
            plt.ylim(0, 90)  # Set the y-axis upper limit
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels diagonally

            # Add a title for the entire figure
            if filename == 'result_summary.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: No\n classifier: random forest')
            elif filename == 'result_summary_PP.csv' and classifier == 'forest':
                plt.suptitle('Post-processing: Yes\n classifier: random forest')
            elif filename == 'result_summary.csv' and classifier == 'extremely':
                plt.suptitle('Post-processing: No\n classifier: extremely randomized trees')
            else:
                plt.suptitle('Post-processing: Yes\n classifier: extremely randomized trees')

            plt.tight_layout()

            plt.show()


if __name__ == '__main__':
    main()
