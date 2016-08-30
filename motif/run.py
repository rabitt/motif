"""Code for running the full pipeline
"""
import numpy as np

from .core import CONTOUR_EXTRACTOR_REGISTRY
from .core import FEATURE_EXTRACTOR_REGISTRY
from .core import CONTOUR_CLASSIFIER_REGISTRY


def process(audio_files=None, training_pairs=None, testing_pairs=None,
            extract_id='salamon', feature_id='bitelli',
            classifier_id='random_forest'):
    contour_extractor = get_extract_module(extract_id)
    feature_extractor = get_features_module(feature_id)
    contour_classifier = get_classify_module(classifier_id)

    if training_pairs is not None:
        X_train, Y_train, train_contours = process_with_labels(
            contour_extractor, feature_extractor, training_pairs
        )
        contour_classifier.fit(X_train, Y_train)

        # get training score
        Y_pred = contour_classifier.predict(X_train)
        train_scores = contour_classifier.score(Y_pred, Y_train)

    if testing_pairs is not None:
        X_test, Y_test, test_contours = process_with_labels(
            contour_extractor, feature_extractor, testing_pairs
        )

        # get testing score
        Y_pred = contour_classifier.predict(X_test)
        test_scores = contour_classifier.score(Y_pred, Y_test)

    if audio_files is not None:
        contour_list = process_audio_only(
            contour_extractor, feature_extractor, contour_classifier,
            audio_files
        )

    return (
        train_contours, train_scores, test_contours, test_scores, contour_list
    )


def process_audio_only(contour_extractor, feature_extractor,
                       contour_classifier, audio_files):

    contour_list = []
    for audio_filepath in audio_files:
        ctr = contour_extractor.compute_contours(audio_filepath)

        X = feature_extractor.compute_features(ctr)
        Y = contour_classifier.predict(X)

        contour_list.append((ctr, X, Y))
    return contour_list


def process_with_labels(contour_extractor, feature_extractor, file_pairs):
    contour_list = []
    features_list = []
    labels_list = []

    for audio_filepath, annotation in file_pairs:

        ctr = contour_extractor.compute_contours(audio_filepath)
        Y_train, _ = ctr.compute_labels(annotation)
        X_train = feature_extractor.compute_features(ctr)

        features_list.append(X_train)
        labels_list.append(Y_train)
        contour_list.append(ctr)

    X = np.concatenate(features_list)
    Y = np.concatenate(labels_list)

    return X, Y, contour_list


def get_extract_module(extract_id):
    """Obtains a configured ContourExtractor given an algorithm identificator.

    Parameters
    ----------
    extract_id : str
        Extract algorithm identificator (e.g., salamon, hll).

    Returns
    -------
    module : object
        Object containing the selected ContourExtractor module.
        None if no extract module is needed.
    """
    if extract_id is None:
        return None
    try:
        module = CONTOUR_EXTRACTOR_REGISTRY[extract_id]
    except AttributeError:
        raise RuntimeError("Algorithm %s can not be found in motif!" %
                           extract_id)
    return module


def get_features_module(feature_id):
    """Obtains a configured ContourFeatures given an algorithm identificator.

    Parameters
    ----------
    feature_id : str
        Feature algorithm identificator (e.g., bitteli, melodia).

    Returns
    -------
    module : object
        Object containing the selected ContourFeatures module.
        None if no extract module is needed.
    """
    if feature_id is None:
        return None
    try:
        module = FEATURE_EXTRACTOR_REGISTRY[feature_id]
    except AttributeError:
        raise RuntimeError("Algorithm %s can not be found in motif!" %
                           feature_id)
    return module


def get_classify_module(classifier_id):
    """Obtains a configured Classifier given an algorithm identificator.

    Parameters
    ----------
    classifier_id : str
        Classifier algorithm identificator (e.g., random_forest, mv_gaussian).

    Returns
    -------
    module : object
        Object containing the selected Classifier module.
        None if no extract module is needed.
    """
    if classifier_id is None:
        return None
    try:
        module = CONTOUR_CLASSIFIER_REGISTRY[classifier_id]
    except AttributeError:
        raise RuntimeError("Algorithm %s can not be found in motif!" %
                           classifier_id)
    return module
