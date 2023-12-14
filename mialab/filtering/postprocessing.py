"""The post-processing module contains classes for image filtering mostly applied after a classification.

Image post-processing aims to alter images such that they depict a desired representation.
"""
import warnings

import numpy as np
#import pydensecrf.densecrf as crf
#import pydensecrf.utils as crf_util
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
import cv2

class ImagePostProcessingParameters(pymia_fltr.FilterParams):
    """Image post-processing parameters."""

    def __init__(self, segmentation: sitk.Image):
        """Initializes a new instance of the ImagePostProcessingParameters

        Args:
            image (sitk.Image): The image.
            img_mask (sitk.Image): The brain mask image.
        """
        self.segmentation = segmentation

class ImagePostProcessing(pymia_fltr.Filter):
    """Represents a post-processing filter."""

    def __init__(self):
        """Initializes a new instance of the ImagePostProcessing class."""
        super().__init__()

    def execute(self, multiclass_image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters.

        Returns:
            sitk.Image: The post-processed image.
        """

        # todo: replace this filter by a post-processing - or do we need post-processing at all?
        multiclass_image = params.segmentation

        # Calculate class volumes
        class_values = np.unique(sitk.GetArrayFromImage(multiclass_image))
        class_volumes = {}

        for class_val in class_values:
            # Count the number of pixels/voxels for each class
            class_mask = sitk.GetArrayFromImage(multiclass_image) == class_val
            volume = np.sum(class_mask)
            class_volumes[class_val] = volume

        # Sort classes based on volumes in descending order to define priority
        sorted_classes = sorted(class_volumes, key=class_volumes.get, reverse=True)

        # Create an empty image to store labeled components
        labeled_components = sitk.Image(multiclass_image.GetSize(), sitk.sitkUInt32)
        labeled_components = sitk.Resample(labeled_components, multiclass_image, sitk.Transform(), sitk.sitkLinear, 0.0, labeled_components.GetPixelID())

        # Label connected components for each class separately
        for class_val in sorted_classes:
            if class_val != 0:
                # Create a binary image for the current class
                binary_class_image = (multiclass_image == class_val)

                # Label connected components for the current class
                connected_components = sitk.ConnectedComponent(binary_class_image)
                sorted_connected_components = sitk.RelabelComponent(connected_components, sortByObjectSize=True)

                largest_component_binary_image = sitk.Image(multiclass_image.GetSize(), sitk.sitkUInt32)
                for i in range(1, 6):
                    largest_component_binary_image = sitk.Cast((sorted_connected_components == i), sitk.sitkUInt32)

                    # Assign labels for the current class to the output image
                    labeled_components += sitk.Mask(labeled_components, largest_component_binary_image) + \
                                         (largest_component_binary_image * class_val)

        return labeled_components

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImagePostProcessing:\n' \
            .format(self=self)

class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask

# class DenseCRFParams(pymia_fltr.FilterParams):
#     """Dense CRF parameters."""
#     def __init__(self, img_t1: sitk.Image, img_t2: sitk.Image, img_proba: sitk.Image):
#         """Initializes a new instance of the DenseCRFParams
#
#         Args:
#             img_t1 (sitk.Image): The T1-weighted image.
#             img_t2 (sitk.Image): The T2-weighted image.
#             img_probability (sitk.Image): The posterior probability image.
#         """
#         self.img_t1 = img_t1
#         self.img_t2 = img_t2
#         self.img_probability = img_probability
#
#
# class DenseCRF(pymia_fltr.Filter):
#     """A dense conditional random field (dCRF).
#
#     Implements the work of Krähenbühl and Koltun, Efficient Inference in Fully Connected CRFs
#     with Gaussian Edge Potentials, 2012. The dCRF code is taken from https://github.com/lucasb-eyer/pydensecrf.
#     """
#
#     def __init__(self):
#         """Initializes a new instance of the DenseCRF class."""
#         super().__init__()
#
#     def execute(self, image: sitk.Image, params: DenseCRFParams = None) -> sitk.Image:
#         """Executes the dCRF regularization.
#
#         Args:
#             image (sitk.Image): The image (unused).
#             params (FilterParams): The parameters.
#
#         Returns:
#             sitk.Image: The filtered image.
#         """
#
#         if params is None:
#             raise ValueError('Parameters are required')
#
#         img_t2 = sitk.GetArrayFromImage(params.img_t1)
#         img_ir = sitk.GetArrayFromImage(params.img_t2)
#         img_probability = sitk.GetArrayFromImage(params.img_probability)
#
#         # some variables
#         x = img_probability.shape[2]
#         y = img_probability.shape[1]
#         z = img_probability.shape[0]
#         no_labels = img_probability.shape[3]
#
#         img_probability = np.rollaxis(img_probability, 3, 0)
#
#         d = crf.DenseCRF(x * y * z, no_labels)  # width, height, nlabels
#         U = crf_util.unary_from_softmax(img_probability)
#         d.setUnaryEnergy(U)
#
#         stack = np.stack([img_t2, img_ir], axis=3)
#
#         # Create the pairwise bilateral term from the above images.
#         # The two `s{dims,chan}` parameters are model hyper-parameters defining
#         # the strength of the location and image content bi-laterals, respectively.
#
#         # higher weight equals stronger
#         pairwise_energy = crf_util.create_pairwise_bilateral(sdims=(1, 1, 1), schan=(1, 1), img=stack, chdim=3)
#
#         # `compat` (Compatibility) is the "strength" of this potential.
#         compat = 10
#         # compat = np.array([1, 1], np.float32)
#         # weight --> lower equals stronger
#         # compat = np.array([[0, 10], [10, 1]], np.float32)
#
#         d.addPairwiseEnergy(pairwise_energy, compat=compat,
#                             kernel=crf.DIAG_KERNEL,
#                             normalization=crf.NORMALIZE_SYMMETRIC)
#
#         # add location only
#         # pairwise_gaussian = crf_util.create_pairwise_gaussian(sdims=(.5,.5,.5), shape=(x, y, z))
#         #
#         # d.addPairwiseEnergy(pairwise_gaussian, compat=.3,
#         #                     kernel=dcrf.DIAG_KERNEL,
#         #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
#
#         # compatibility, kernel and normalization
#         Q_unary = d.inference(10)
#         # Q_unary, tmp1, tmp2 = d.startInference()
#         #
#         # for _ in range(10):
#         #     d.stepInference(Q_unary, tmp1, tmp2)
#         #     print(d.klDivergence(Q_unary) / (z* y*x))
#         # kl2 = d.klDivergence(Q_unary) / (z* y*x)
#
#         # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
#         map_soln_unary = np.argmax(Q_unary, axis=0)
#         map_soln_unary = map_soln_unary.reshape((z, y, x))
#         map_soln_unary = map_soln_unary.astype(np.uint8)  # convert to uint8 from int64
#         # Saving int64 with SimpleITK corrupts the file for Windows, i.e. opening it raises an ITK error:
#         # Unknown component type error: 0
#
#         img_out = sitk.GetImageFromArray(map_soln_unary)
#         img_out.CopyInformation(params.img_t1)
#         return img_out
