"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load
np.random.seed(1)

def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': False,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    # labels_train is your array of class labels
    unique_labels, counts = np.unique(labels_train, return_counts=True)

    # Calculate the total number of instances
    total_instances = len(labels_train)

    rmin = 1.0
    rmax = 15.12
    tmin = 1.0
    tmax = 1.5

    # Print the count and percentage for each class
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_instances)
        classweight = (29958 / count)
        normalized_weights = ((classweight - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin

        print(
            f"Class {label}: {count} data | ({percentage:.2f}%) percentage | ({classweight:.2f}) class weight | ({normalized_weights:.2f}) normalized weight")

    selected_classifier = 'extremely'
    class_weights = {1: 1, 2: 1.25, 3: 5.54, 4: 15.12, 5: 5.46}

    if selected_classifier == 'forest':
        classifier = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                        n_estimators=500,
                                                        max_depth=150,
                                                        class_weight=class_weights)

    elif selected_classifier == 'extremely':
        classifier = sk_ensemble.ExtraTreesClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                      n_estimators=100,
                                                      max_depth=25,
                                                      class_weight=None)

    start_time = timeit.default_timer()
    classifier.fit(data_train, labels_train)
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        predictions = classifier.predict(img.feature_matrix[0])
        probabilities = classifier.predict_proba(img.feature_matrix[0])
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    post_process_params = {'simple_post': True}

    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=True)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')

    # generate result directory, if it does not exists
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)
    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # Create result_summary.csv for '_SEG.mha' files
    result_summary_file = os.path.join(result_dir, 'result_summary.csv')

    # Create result_summary_PP.csv for '_SEG-PP.mha' files
    result_summary_pp_file = os.path.join(result_dir, 'result_summary_PP.csv')

    evaluator.results.clear()

    # Evaluate and save results for segmentation images
    for i, img in enumerate(images_test):
        # Evaluate segmentation without post-processing
        evaluator.evaluate(images_prediction[i], img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        # Save results for segmentation images
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)

    # Write statistics for segmentation images to CSV
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nIndividual statistic results for segmentation images...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # Clear results for post-processed images
    evaluator.results.clear()

    # Evaluate and save results for segmentation images
    for i, img in enumerate(images_test):
        # Evaluate segmentation without post-processing
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        # Save results for segmentation images
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)

    # Write statistics for segmentation images to CSV
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_pp_file, functions=functions).write(evaluator.results)
    print('\nIndividual statistic results for segmentation images...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
