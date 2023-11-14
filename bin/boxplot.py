import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')

    directory_path = r'..\bin\mia-result'

    result_name = '2023-11-10-18-34-11'

    file_path = os.path.join(directory_path, result_name, 'results.csv')

    df = pd.read_csv(file_path, delimiter=';')

    plt.figure(figsize=(10, 6))
    plt.title('Dice Coefficients per Label')
    plt.xlabel('Label')
    plt.ylabel('Dice Coefficient')
    plt.boxplot([df[df['LABEL'] == label]['DICE'] for label in df['LABEL'].unique()],
                labels=df['LABEL'].unique())
    plt.show()


if __name__ == '__main__':
    main()
