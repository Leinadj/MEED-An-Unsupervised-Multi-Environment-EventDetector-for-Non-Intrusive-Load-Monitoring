import platform
import matplotlib
if platform.system() == "Linux": #for matplotlib on Linux
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.cluster import DBSCAN
from sklearn import utils
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
import tensorflow as tf
import pdb
import warnings
import time
import math
from sklearn import preprocessing
from datetime import timedelta
from scipy.signal import argrelextrema
import scipy
from datetime import datetime
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, UpSampling1D, Cropping1D, ZeroPadding1D
from keras.models import load_model
from keras import optimizers
from keras import backend as kback
import os

class EventDet_MEED(BaseEstimator, ClassifierMixin):
    """
    MEED Event Detector according the paper of Daniel Jorde

    Attributes
    ----------
    model_location_p : str
        Either this is a path to an existing model in the Keras model format,
        or in the case a new model will be trained, it will be stored using the model_location path.
        This path should already include the name of the model.

    coarse_mse_threshold_p : int
        The threshold that is applied to the MSE reconstruction error of the autoencoder.
        Determine whether a coarse event (i.e. event window) is detected or not.

    signal_length_p : int, optional (default = 100)
        Length of the input signal.
        This is also the length the model is expecting at its input nodes and the length of the output of the model.

    min_time_between_events_threshold_p : int, optional (default = 3)
        The minimum number of samples that is maintained between two consecutive event candidates.
        It is used in the fine grained event detection step.
        Use this parameter to incorporate a handmade event definition into the algorithm.

    fluctuation_limit_diff_threshold_p : int, optional (default = 1)
        Is used to suppress minor fluctations in the signal.

    rms_periods_p : int, optional (default = 5)
        The number of periods the Root-Mean-Square (RMS) of the current signal is computed over.

    network_frequency_p: int, optional (default = 50)
        The base frequency of the power network, where the data is recorded in.
        It is measured in Hz.
        For example: Europe = 50 (Hz) or US = 60 (Hz)

    window_size_sec_p : int, optional (default = 10)
        Size of the input window

    Functions
    ---------
    fit

    predict

    score: static

    _train_LSTM_autoencoder

    _coarse_detection_step

    _fine_grained_detection_step

    _compute_timestamp_from_index


    """

    def __init__(self, model_location_p, coarse_mse_threshold_p, signal_length_p=100,
                 min_time_between_events_threshold_p=3,
                 fluctuation_limit_diff_threshold_p=1, rms_periods_p=5, network_frequency_p=50, window_size_sec_p=10):
        """
        Parameters
        ----------
        model_location_p : str
            Either this is a path to an existing model in the Keras model format,
            or in the case a new model will be trained, it will be stored using the model_location path.
            This path should already include the name of the model.

        coarse_mse_threshold_p : int
            The threshold that is applied to the MSE reconstruction error of the autoencoder.
            Determine whether a coarse event (i.e. event window) is detected or not.

        signal_length_p : int, optional (default = 100)
            Length of the input signal.
            This is also the length the model is expecting at its input nodes and the length of the output of the model.

        min_time_between_events_threshold_p : int, optional (default = 3)
            The minimum number of samples that is maintained between two consecutive event candidates.
            It is used in the fine grained event detection step.
            Use this parameter to incorporate a handmade event definition into the algorithm.

        fluctuation_limit_diff_threshold_p : int, optional (default = 1)
            Is used to suppress minor fluctations in the signal.

        rms_periods_p : int, optional (default = 5)
            The number of periods the Root-Mean-Square (RMS) of the current signal is computed over.

        network_frequency_p : int, optional (default = 50)
            The base frequency of the power network, where the data is recorded in.
            It is measured in Hz.
            For example: Europe = 50 (Hz) or US = 60 (Hz)

        window_size_sec_p : int, optional (default = 10)
            Size of the input window

        Returns
        -------

        None

        """
        self.model_location = model_location_p
        self.coarse_mse_threshold = coarse_mse_threshold_p
        self.signal_length = signal_length_p
        self.min_time_between_events_threshold = min_time_between_events_threshold_p
        self.fluctuation_limit_diff_threshold = fluctuation_limit_diff_threshold_p
        self.rms_periods = rms_periods_p
        self.network_frequency = network_frequency_p
        self.window_size_sec = window_size_sec_p

    def fit(self, train_new_model_p=False, use_default_model_p=None):
        """
        Fit the model by either training a new autoencoder or by loading an existing model.

        Parameters
        ----------
        train_new_model_p : boolean, optional (default = False)
            If True : a new model is trained using the _train_LSTM_autoencoder method

        use_default_model_p : boolean, optional (default = None)
            If True and If traine_new_model is False: No model is loaded from disk, but the default graph in memory
            is used.
            It uses the tensorflow.get_default_graph() and tensorflow.get_default_session() under the hood.

        Returns
        -------
            self
        """

        if train_new_model_p == False:
            if use_default_model_p is None:
                # reset default graph to ensure nothing is left to interfer with the new graph
                tf.reset_default_graph()
                kback.clear_session()
                self.model = load_model(self.model_location)  # Load the Model from Disk
                self.model._make_predict_function()  # build and compile the function explictily on the GPU
                self.sess = tf.Session()  # Create a tensorflow session
                self.sess.run(tf.global_variables_initializer())  # initialize all the parameter values
                self.default_graph = tf.get_default_graph()  # get the default graph, hence, our current graph
                self.default_graph.finalize()  # freeze the graph, can not be altered anymore
            # Uses an already existing default graph in the case the use_default_graph parameter is set to false

            else:  # use the default graph that is already on memory
                self.default_graph = tf.get_default_graph()
                self.sess = tf.get_default_session()
                self.model = use_default_model_p

        elif train_new_model_p == True:
            # Evaluate if this is all necessary or not
            self.model = self._train_LSTM_autoencoder()
            self.model._make_predict_function()  # build and compile the function explicitly on the GPU
            self.sess = tf.Session()  # Create a Tensorflow session
            self.sess.run(tf.global_variables_initializer())  # initialize all the parameter values
            self.default_graph = tf.get_default_graph()  # get the default graph, hence, our current graph
            self.default_graph.finalize()  # freeze the graph, can not be altered anymore

        else:
            raise ValueError("train_new_model is a boolean parameter!")

        self.is_fitted = True

        # get the output shape shape of the model
        # it equals the input shape, as it is an autoencoder model that use the MSE reconstruction error
        self.model_output_shape = self.model.layers[3].get_output_at(0).get_shape().as_list()

        if self.model_output_shape[
            1] != self.signal_length:  # the output ahs a shape of (batch, signal_length, 1) where 1 is the number of channels
            raise ValueError("The signal length has to match the length of the signal the model expects")
        return self

    def predict(self, current_cumsum_p, current_rms_p, start_datetime_p, use_median_p=False, return_MSE_p=False, cpu_only_p=True):
        """
        Predicts the event timestamp for a given input signal.
        Internally it calls the _coarse_detection_step method and if an event is detected here, it
        also calls the _fined_grained_detection_step method

        Parameters
        ----------
        current_cumsum_p : ndarray, shape = [signal_length_p]
            The Cumulative Sum (CUMSUM) over the Root-Mean-Square (RMS) values of the current signal.
            Can be computed using the compute_input_signal functions.

        current_rms_p : ndarray, shape = [signal_length_p]
            The Root-Mean-Square (RMS) values of the current signal.
            Can be computed using the compute_input_signal functions.

        start_datetime_p : datetime
            The start timestamp of the current signal window

        use_median_p : boolean, optional (default = False)
            If True: The _fine_grained_event_detection_step uses the median of the signal window to supress flucutations.
            If False: The _fine_grained_event_detection_step uses the mean of the signal window to supress flucutations.

        return_MSE_p : boolean
            If true, a list of MSE values is returned. This list contains more values
            then the change_points_timestamps_list, as also the mse values of the TN
            are returned

        cpu_only_p : boolean
            Use only cpus for predicting

        Returns
        -------
        timestamps : list
            List of datetime objects where events were detected in the current window

        mse : float, optional
            If return_MSE_p is set to true, also the mse_list is returned
        """

        if cpu_only_p is True:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        check_is_fitted(self, ["is_fitted"])  # check if fit() was called before

        # if the window is empty return empty list
        if len(current_cumsum_p) != self.signal_length:
            raise ValueError(
                "The CUMSUM input values do not have the length that was specified in the signal_length parameter!"
                "It may be the case, that the input was empty! Please check your input features computation!")

        if len(current_rms_p) != self.signal_length:
            raise ValueError(
                "The RMS input values do not have the length that was specified in the signal_length parameter!"
                "It may be the case, that the input was empty! Please check your input features computation!")

        if return_MSE_p == True:
            # Call the Coarse event detection method
            coarse_event_detected, mse = self._coarse_detection_step(current_cumsum_p=current_cumsum_p,
                                                                     return_MSE_p=return_MSE_p)
        else:
            coarse_event_detected = self._coarse_detection_step(current_cumsum_p=current_cumsum_p,
                                                                return_MSE_p=return_MSE_p)

        if coarse_event_detected == True:  # A coarse event was detected

            # Call the Fine Grained Event Detection Step
            event_timestamps = self._find_grained_detection_step(current_rms_p=current_rms_p,
                                                                 start_datetime_p=start_datetime_p,
                                                                 use_median_p=use_median_p)


        else:  # No coarse event was detected
            event_timestamps = []

        if return_MSE_p == False:
            return event_timestamps
        else:
            return event_timestamps, mse

    def compute_input_signal(self, current_p, period_length_p):
        """
        Compute Root-Mean-Square(RMS) and the Cumulative Sum (CUMSUM) of the RMS values off the input current signal.
        These values are computed over a multiple of the period_length_p (in samples).
        It multiplies the value stored in the rms_periods_p parameter with the period_length_p to compute the output.

        Parameters
        ----------
        current_p : ndarray, shape = [signal_length_p]
            The raw current signal.

        period_length_p : int
            The number of samples that are in one period of the signal

        Returns
        -------
        current_cumsum : ndarray, shape = [signal_length_p]
            The CUMSUM over the RMS values

        current_rms : ndarray, shape = [signal_length_p]
            The RMS values over the raw current_p signal.

           """

        if len(current_p) == 0:
            raise ValueError("The current signal you have provided, is empty! Please check your input!")

        metrics = Electrical_Metrics()

        current_rms = metrics.compute_single_rms(current_p, period_length=int(
            period_length_p * self.rms_periods))  # Preprocessing Function for the Stream
        # CUMSUM
        differences_np = current_rms - np.mean(current_rms)
        current_cumsum = np.cumsum(differences_np)

        if len(current_cumsum) != self.signal_length or len(current_rms) != self.signal_length:
            raise ValueError(
                "The signal length specified is not in line with the period_length and the size of the current signal "
                "that is provided! Please check your window size, sampling rate and other relevant "
                "parameters!")

        return current_cumsum, current_rms

    @staticmethod
    def score(ground_truth_events_p, detected_events_p, number_of_samples_in_dataset_p=None,
              tolerance_limit_sec_p=1, return_event_lists_p=False):
        """
        This function can be used to determined the values of the confusion matrix.

        It works as follows:
        For each of the detected events, we build an interval around the detected timestamp using the tolerance_limit_sec
            Then we check for each of the detected intervals if we find a ground truth event that falls within this intervall
             If this is the case we have found a True Positive Event (TP) :
               1. TP += 1
               2. Eliminate this event from the ground truth list, in order to: avoid double detection
               In the case there is more then one ground truth events that fits into the interval, we take the first one
             If this is not the case we have found a False Positive Event (FP):
               1. FP += 1

             After this procedure: all events that are left in the ground truth list have not been detected by the event detection algorithm
             (because they were not removed from the in the previous step)
             Hence those events are actually true, but are delcared to be false by the algorithm: They are False Negatives (FN)
             For each of those events left in the ground truth list: FN += 1

             Finally we determine the number of True Negative Events(TN):
             Lastly, when all the detected and groundtruth events have been processed, the TN are calculated by
             subtracting the TP, FN, and FP from the number of samples in the dataset, i.e., all the positions where an event could have happened
             The start and the end time of the dataset and the information about sampling rate is used to compute the total
             number of possible event timestamps.

        Parameters
        ----------
        ground_truth_events_p : list
            List of datetime objects, containing the ground truth event timestamps.

        detected_events_p : list
            List of datetime objects, containing the event detected by the event detector

        number_of_samples_in_dataset_p : int
            Number of samples, i.e. number of possible event positions in the dataset.
            When computing period wise features, like RMS, values use the corresponding number of samples.
            The value is used to compute the number of True Negative Events.

        tolerance_limit_sec_p : int, optional (default = 1)
            The tolerance limit that is used to determine the scores of the confusion matrix.

        return_events_list_p : boolean, optional (default = False)
            If set to True, the events that are TP, FP, and FN are returned in the results dictionary

        Returns
        -------
        if grid_search_mode is set to True, only the f1score is returned, otherwise:

        results: dictionary
            A dictionary containing the following keys:  tp, fp, fn, tn, f1score, recall, precision, fpp, fpr
            With being:
            tp = True Positives
            fp = False Positives
            fn = False Negatives
            tn = True Negatives
            f1score = F1-Score Metric
            recall = Recall Metric
            precision = Precision Metric
            fpp = False Positive Percentage Metric
            fpr = False Positive Rate Metric

            If return_events_list_p is set to True, the results also contain the following keys:
            tn_events = events that are tn
            fp_events = events that are fp
            fn_events = events that are fn

        """

        ground_truth_events = ground_truth_events_p  # ground_truth_events_p.copy() #copy the events
        detected_events = detected_events_p.copy()  # copy the events

        results = {}
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        TP_events = []
        FP_events = []
        FN_events = []

        # Compute the Tolerance Intervals around the detected events and iterate through the detected events
        for detected_event in detected_events:
            upper_limit = detected_event + timedelta(seconds=tolerance_limit_sec_p)
            lower_limit = detected_event - timedelta(seconds=tolerance_limit_sec_p)

            # Check which ground_truth events lie within this interval
            matching_ground_truth_events = [(index, ground_truth_event) for index, ground_truth_event in
                                            enumerate(ground_truth_events) if
                                            lower_limit <= ground_truth_event <= upper_limit]

            if len(matching_ground_truth_events) > 0:  # at least (minimum) one ground truth event was detected
                TP += 1
                first_index = matching_ground_truth_events[0][0]

                TP_events.append(ground_truth_events[first_index])
                # It can be possible that there are multiple ground truth events within this interval, hence,
                # we only consider the first one and remove it from the list
                del ground_truth_events[first_index]

            else:  # No Matching Event was found
                FP_events.append(detected_event)
                FP += 1

        # Use the remaining ground truth events to determine the number of FN
        FN = len(ground_truth_events)
        FN_events = ground_truth_events.copy()

        # Use the number of samples in the dataset to determine the TNs in the dataset
        # For BLUED we have data from 20/10/2011 11:58:32.623499 to 27/10/2011 14:10:40.0

        TN = number_of_samples_in_dataset_p - (TP + FP + FN)
        # Compute the final scores
        fpp = FP / (TP + FN)
        fpr = FP / (FP + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1score = (2 * precision * recall) / (precision + recall)

        results["tp"] = TP
        results["fp"] = FP
        results["fn"] = FN
        results["tn"] = TN
        results["f1score"] = f1score
        results["recall"] = recall
        results["precision"] = precision
        results["fpp"] = fpp
        results["fpr"] = fpr

        if return_event_lists_p is True:
            results["tp_events"] = TP_events
            results["fp_events"] = FP_events
            results["fn_events"] = FN_events

        return results

    def _train_LSTM_autoencoder(self, merge_mode_p="concat", optimizer_p=optimizers.Adam()):
        """
        Function to train the autoencoder model, stores the model under the model_save_path parameter
        Returns:


        Parameters
        ----------
        merge_mode_p : str, optional (default="concat")
            Concatenation mode used to merge the forward and backward LSTM.
        optimizer_p : keras.optimizers object, optional, (default=keras.optimizers.Adam())
            Optimizer used for the autoencoder

        Returns
        -------

        """

        model = Sequential()

        # Add the three Layers
        # One LSTM cell in each direction
        model.add(Bidirectional(LSTM(216, return_sequences=True), input_shape=(int(self.signal_length), 1),
                                merge_mode=merge_mode_p))
        model.add(Bidirectional(LSTM(108, return_sequences=True), input_shape=(216, 1), merge_mode=merge_mode_p))
        model.add(Bidirectional(LSTM(216, return_sequences=True), input_shape=(108, 1), merge_mode=merge_mode_p))

        # Add the output Layer: Inputs are normalized between 0 and 1 --> Softmax
        # This adds a dense layer as the output, it outputs one value at a timestep
        model.add(TimeDistributed(Dense(1, activation='linear')))

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer=optimizer_p, metrics=["mse"])  #

        return model

    def _coarse_detection_step(self, current_cumsum_p, return_MSE_p):
        """
        This function uses the autoencoder model to perform the coarse event detection step.
        It applies values stored in the coarse_mse_threshold_p to the Mean-Squared-Error (MSE) of the input window
        and the predicted window.

        Parameters
        ----------
        current_cumsum_p : ndarray, shape = [signal_length_p]
            The CUMSUM of the current RMS signal.

        return_MSE_p : boolean
            If True, also returns the MSE value

        Returns
        ------
        event_detected : boolean
            If True: an event was detected
            If False: no event was detected

        mse: float, optional
            Is returned if return_MSE_p is set True

        """

        # Adapt the shape of the input for the Coarse Event Detection Step with the Autoenoder
        # The LSTM Model expects the input with the shape: (window, signal, channel)
        # This event detector only allows one channel at a time hence the shape is set to  (1, signal_length_p, 1)
        current_cumsum_p = np.reshape(current_cumsum_p,
                                      (1, self.signal_length, 1))  # reshape to (window,channel,signal)

        # Reconstruct the Input with the Autoencoder
        with self.default_graph.as_default():
            sample_prediction = self.model.predict(current_cumsum_p)
        # flatten the sample prediction
        sample_prediction = sample_prediction.flatten()  # should be same shape  as the current_cumsum_p input
        current_cumsum_p = current_cumsum_p.flatten()  # reflatten the input again

        if current_cumsum_p.shape != sample_prediction.shape:
            raise ValueError("The shape of the input is different compared to the shape of the prediction!")

        # Compute the mse
        mse = np.mean(np.power(current_cumsum_p - sample_prediction, 2), axis=None)

        if mse > self.coarse_mse_threshold:  # Coarse event detected

            if return_MSE_p == False:
                return True
            else:
                return True, mse
        else:  # No coarse event detected
            if return_MSE_p == False:
                return False
            else:
                return False, mse

    def _find_grained_detection_step(self, current_rms_p, start_datetime_p, use_median_p):
        """
        This function exectutes the fine-grained event detection steps.
        In case a coarse event was detected, it is executed.
        This function aims to identify the correct number and exact location of events in the window.

        Parameters
        ----------
        current_rms_p : ndarray, shape = [signal_length_p]
            The RMS values over the current signal.

        start_datetime_p : datetime
            The start time of the current window.

        use_median_p : boolean
            If True: The _fine_grained_event_detection_step uses the median of the signal window to supress flucutations.
            If False: The _fine_grained_event_detection_step uses the mean of the signal window to supress flucutations.

        Returns
        -------
        change_points_timestamps : list
            List of datetime objects, with the detected event timestamps

        """

        # 1. Binarize the signal
        if use_median_p == False:
            mean_limit = np.mean(current_rms_p)  # the mean value is the limit use to binary the signal
        else:  # use the median as a limit
            mean_limit = np.median(current_rms_p)

        binarized_input = np.array(current_rms_p)
        small_mask = current_rms_p < mean_limit
        big_mask = current_rms_p > mean_limit
        binarized_input[small_mask] = 0
        binarized_input[big_mask] = 1

        # 2.1 check for changes in the binary signal --> when it changes from 0 to 1 or from 1 to 0 something happens
        comparision_indices = np.where(binarized_input[:-1] != binarized_input[1:])[0]  # finds all these change points
        last_index = len(binarized_input) - 1  # get the last index in the signal to append it to the changes
        comparision_indices = np.append(comparision_indices,
                                        last_index)  # append the last element of the current to the indices --> to also get the events at the end of the signal
        # we append the last index, because we want to take the diff of the last event index too. If it is to close to the end of the window we discard it!

        # 2.2 Compute the difference = distance between those indices --> to filter out minor flucutations
        indices_diff = np.diff(comparision_indices)  # one element less due to the difference operation
        indices_diff_relevant = np.where(
            indices_diff >= self.min_time_between_events_threshold)  # get the indices, where the change is at least 5 values long

        relevant_comparision_indices = comparision_indices[
            indices_diff_relevant]  # indices that are zero, before their change to 1

        relevant_comparision_indices = relevant_comparision_indices + 1  # add one to the indices to adjust this
        # Now we have all the change indices from the binarized sindal that maintain a minimum distance of "min_time_between_events_threshold" between each other

        # Postprocessing of the events to additionally reduce the amount of false positives
        # The two possibilities that can occur are:
        # 1. the fine grained step before has detected min. 1 event --> take those
        # 2. the fine grained step failed to do so, but despite this the AE MSE error is above the threshold --> take the most extreme value in the current as the event

        if len(
                relevant_comparision_indices) > 0:  # if the fine grained event detector detected an event --> preprocess them and extract relevant features
            change_points = []

            if relevant_comparision_indices[-1] <= len(binarized_input):
                change_points.append(relevant_comparision_indices)
            else:
                relevant_comparision_indices = relevant_comparision_indices - 1
                change_points.append(relevant_comparision_indices)

            # COMPUTE THE CHANGE POINTS USING THE LIMIT
            change_points = np.array(change_points).flatten()

            last_index = len(current_rms_p) - 1
            last_index = int(last_index)
            change_points_temp = list(change_points)
            change_points_temp.append(last_index)  # append last index to the change_points
            change_point_values = []
            for index, cp in enumerate(change_points_temp):
                if cp != last_index:
                    nxt_idx = index + 1
                    next_cp = change_points_temp[int(nxt_idx)]

                    # most extreme value: find maximum and minimum value,
                    # change_point_values.append(max(np.abs(np.max(current_rms_p[cp:next_cp])), np.abs(np.min(current_rms_p[cp:next_cp]))))
                    change_point_values.append(np.mean(current_rms_p[cp:next_cp]))

            change_point_values = np.array(change_point_values)

            # change_point_values = current_rms_p[change_points]

            # Check Whether the Change Point values are just minor fluctuations above the mean of the signal (the limit)
            change_point_values_limit_diffs = change_point_values - mean_limit  # limit = mean of the signal window
            change_point_values_limit_diffs_abs = np.abs(change_point_values_limit_diffs)

            # Select the indices of the change points where the events are not just minor flucutations above the mean
            change_point_values_limit_diffs_abs_treshold_indices = \
            np.where(change_point_values_limit_diffs_abs > self.fluctuation_limit_diff_threshold)[
                0]  # the true ones = index 0 in the return tuple

            if len(
                    change_point_values_limit_diffs_abs_treshold_indices) > 0:  # if there exists a change point above the limit
                change_point_values = change_point_values[
                    change_point_values_limit_diffs_abs_treshold_indices]  # select the indices
                change_points = change_points[change_point_values_limit_diffs_abs_treshold_indices]

            else:  # take the point with the maximum absolute difference to the limit (the limit is the mean of the signal)
                change_point_values_limit_diffs_abs_max_idx = np.argmax(change_point_values_limit_diffs_abs)
                change_points = np.array([change_points[change_point_values_limit_diffs_abs_max_idx]]).flatten()


        else:  # if the fine grained event detector did not find an event --> take the most extreme value in the window as the event
            change_points = []
            change_point_values = []

            # Take the maximum value in the signal, the autoencoder is the truth!
            maxm = argrelextrema(current_rms_p, np.greater)  # indices
            minm = argrelextrema(current_rms_p, np.less)  # indices

            max_current_idx = np.argmax(current_rms_p[maxm])
            min_current_idx = np.argmax(current_rms_p[minm])
            max_current = current_rms_p[max_current_idx]
            min_current = current_rms_p[min_current_idx]

            if np.abs(max_current) >= np.abs(min_current):
                change_points.append(max_current_idx)
                change_point_values.append(max_current)
            else:
                change_points.append(min_current_idx)
                change_point_values.append(min_current)

        # Now Compute the timestamps for the detected change_point indices
        change_points_timestamps = self._compute_timestamp_from_index(indices_p=change_points,
                                                                      start_datetime_p=start_datetime_p)
        return change_points_timestamps

    def _compute_timestamp_from_index(self, indices_p, start_datetime_p):
        """
        This function computes the timestamps from the event indices that are detected.
        It uses the values stored in rms_periods_p and network_frequency_p to do so.

        There are network_frequency_p periods per second
        If the rms is computed over self.rms_periods there are only network_frequency_p / rms_periods values_p / second
        We want the seconds / value, hence we invert the computation and we receive seconds_per_value = rms_periods_p / networkf_requency_p
        Example with network_frequency_p = 60 and rms_periods_p = 5
        There are 60 / 5 = 12 values / second
        We are interested in the seconds / value, hence we invert it and compute 1/12 or 5/60, what is rms_periods_p / network_frequency_p
        For example: 60 Hz network frequency, and rms over 5 periods -->  60 / 5 = 12 periods per second.

        Parameters
        ----------
        indices_p : list
            List containing the indices where events were detected in the current window
        start_datetime_p : datetime
            The start time of the current window.


        Returns
        -------
        timestamps : list
            List of datetime objects, with the detected event timestamps

        """

        seconds_per_value = self.rms_periods / self.network_frequency

        # seconds_of_one_rms_value = self.rms_periods / self.network_frequency
        change_points_seconds = indices_p * seconds_per_value  # sample * (sec / sample) = sec

        timestamps = np.array([start_datetime_p + timedelta(seconds=sec) for sec in
                               change_points_seconds]).flatten()

        return timestamps

class EventDet_Jin(BaseEstimator, ClassifierMixin):
    """
    Reference implementation for the following Event Detection algorithm:
        Robust adaptive Event Detection in Non-Intrusive Load Monitoring
        for Energy Aware Smart Facitlities
        by: Yuanwei Jin, Eniye Tebekaemi, Mario berges, Lucio Soibelman
        link to paper: https://ieeexplore.ieee.org/document/5947314/


    The authors themselves do not provide proper default values:
    But the algorithm was used by Yang et al in their paper:
    Comperative Study of Event Detection Methods for Nonintrusive Appliance Load Monitoring
    --> we take the default parameters from their evaluation

    The algorithm implementation follows the sklearn API

    Short Description of the algorithm:

        Input: Required shape (2, window_size_n)
            - the first array in the input is the pre-event-window
            - the second array in the input is the detection_window
            - each of those windows has size window_size_n
            - they must be equally sized
        Goodness of Fit (GOF) Based Algorithm

        Two Main steps:
        1. Detect an event within each window
        2. Locate the time-instant of change for the event

        Formulate everything as a Binary Hypothesis Test
        Pre-event window distribution: G(x)  => reference distribution
        Detection window distribution: F(x)


        H0: G(x) = F(x)
        H1: G(x) != F(x)


        Reject HO --> reject that the distributions are equal, i.e. there
        is an event if: the x2 chi-squared GOF test is bigger then the
        chi-squared threshold with n-1 degrees of freedom and a confidence level
        The confidence_level should be already pre-computed: (1 - alpha)

        both windows have samples that are iid

        ASSUMPTION 1: The authors only mention the use of a sliding window, not of overlapping windows.
        Furthermore, they do not talk about how they resolve multiple detections of the same event as
        they would need to when using overlapping windows. Hence, we assume that no overlapping windows are used

        ASSUMPTION 2: In addition to ASSUMPTION 1, the authors do not give information on how they determine the exact event timestamps
        As this is critical for the evaluation of the algorithm and for applications relying on the events,
        we do the following: In case the GOF test detects an event, we take the index of the most exteme value (minima or maxima)
        in the detection window as the exact event point


    """
    def __init__(self,window_size_n=10,confidence_level=0.95,alpha=0.01,E=30,minimum_transient_length_s=5,network_frequency=60):
        """
        Initialize the Function
        Default: use the value for the confidence level that is provided by the respective parameter
        If alpha is given, the confidence level is set to 1-alpha

        Parameters
        ----------
        window_size_n : int, optional (default=10)
            Window size used
        confidence_level : float, optional (default=0.95)
            Confidence level for the goodness-of-fit test: 1-alpha
        alpha : float, optional (default=0.01)
            Alpha error for the goodness-of-fit-test
        E : int, optional (default=30)
            Minimum amplitude of detectable jumps in the power signal
        minimum_transient_length_s : int, optional (default=5)
            Minimum length of detectable transient segments
        network_frequency : int, optional (default=60)
            Base frequency of the network the dataset was recorded in
        """

        if not confidence_level > 0 or not confidence_level <= 1:
            raise ValueError("The confidence level must be a value between 0 and 1")

        self.network_frequency = network_frequency

        self.window_size_n = window_size_n
        self.confidence_level = confidence_level

        self.E = E

        if alpha is not None:
            if not alpha > 0 or not alpha <= 1:
                raise ValueError("The value of alpha must be a value between 0 and 1")

            self.confidence_level = 1 - alpha
            self.alpha = alpha
        else:
            self.alpha = 1-confidence_level


        self.fitted_= False

        self.minimum_transient_length_s = minimum_transient_length_s

    def fit(self, x, sampling_rate):

        """

        Used to determine the proper window size n?

        Guideline to do so:

        Model of the discrete average power singal: x_i = e_i + w_i
            w_i: disturbance in measurement, dist. as white gaussian noise (assumed)
            e_i: event indicator

        We want to estimate the Gaussian Process underlying w_i:
            - 1. Estimate sample mean from training data
            - 2. Estimate sample deviation from training data

            define n0 = square(zalpha/2 * sigma_w )/ E)
                with zalpha / 2 : 100 alpha/2 percentage point of the standard normal dist.
                E is user set: 30: exclude events that are less then 30 Watt jumps

        The Maximum window size n1 is limited by the maximum length of the state-transient transient of the signatures : n0 < n <n1

        #NOTE ASSUMPTION: we preset n1 to be 5 seconds for a transient --> therefore we get limits for the window size
        #NOTE we then just take the mean value as the final window size for the classifier

        The function sets the attribute n_zero_ to the minimum window length
        Args:
            X : pre-event training data
            y : labels, always None here

        Returns:
            self

        Parameters
        ----------
        x : ndarrary
            Training data, computed by the compute_input_signal function

        sampling_rate : int
            Sampling rate the dataset was sampled with

        Returns
        -------

        """



        x = np.array(x).flatten()

        sample_stdev = np.std(x) #standard deviation

        upper_percentile = 1 - (self.alpha / 2) #upper alpha/2 percentile
        z_alpha = stats.norm.ppf(upper_percentile)

        n_zero = np.square(((z_alpha / 2) * sample_stdev) / self.E) #lower bound for the windows

        n_one = self.minimum_transient_length_s * sampling_rate #upper bound for the windows

        self.n_one_ = n_one
        self.n_zero_ = n_zero

        try:
            self.n_mean_ = int(np.floor(np.mean([n_one,n_zero])))
        except:
            pdb.set_trace()
        self.is_fitted = True

        return self

    def predict(self, x, det_start_timestamp, det_end_timestamp, mean_window=True):
        """
        Used to predict a input window (pre-event and detection window)

        The window size n is saved in the parameter n_mean_ after fit() is called
        It can be used to feed the data appropriately


        Parameters
        ----------
        x : ndarray
            Pre-event and detection window with shape (2,window_size_n)
        det_start_timestamp : datetime.datetime
            Datetime for the start of the detection_window
        det_end_timestamp : datetime.datetime
            Datetime for the end of the detection_window
        mean_window : boolean, optional (default=True)
            Use the mean value that is calculated in the fit() function

        Returns
        -------
        events : list
            Lists of the events (indices) in the detection window

        """


        check_is_fitted(self, ["is_fitted"])  # check if fit() was called before

        if mean_window == True:
            self.window_size_n = self.n_mean_

        if not isinstance(x, np.ndarray):  # check if the input is an numpy array
            x = np.array(x)

        if not x.ndim == 2:
            raise ValueError("The provided input must be two dimensional. it must be of shape (2,window_size_n)")

        if not np.shape(x)[0] == 2:
            raise ValueError("Only two windows can be handed to the function, i.e. the pre-event and the detection window")


        #Get the windows
        pre_event_window = x[0]
        detection_window = x[1]

        if len(pre_event_window) != len(detection_window):
            raise ValueError("Windows need to have the same shape")
        # Find the decision threshold in the distribution function

        decision_threshold = stats.chi2.ppf(q=self.confidence_level,  # Find the critical value for 95% confidence*
                                            df=(len(detection_window) - 1))  # Df = number of variable categories - 1

        l_gof = self._compute_GOF(pre_event_window, detection_window)

        events = []
        if l_gof > decision_threshold: #compare the statistic with the decision threshold

            #ASSUMPTION 2
            #Determine the exact timestamp of the event, according to ASSUMPTION 2

            maxima = scipy.signal.argrelextrema(detection_window, np.greater)[0] #returns the first index
            if len(maxima) == 0:
                maxima = 0
            else:
                maxima = maxima[0]

            minima = scipy.signal.argrelextrema(detection_window, np.less)[0]
            if len(minima) == 0:
                minima = 0
            else:
                minima = minima[0]

            event_index = None
            #determine which extreme value is more extreme
            if np.abs(detection_window[maxima]) > np.abs(detection_window[minima]):
                event_index = maxima
            else:
                event_index = minima

            event_seconds = event_index * (1 / self.network_frequency)
            # compute the corresponding event timestamp
            event_timestamp = det_start_timestamp + timedelta(seconds=event_seconds)
            events = [event_timestamp]


        else:
            events = []


        return events

    def compute_input_signal(self, voltage, current, period_length):
        """

        Parameters
        ----------
        voltage : ndarray
            Voltage signal, flat array
        current : ndarray
            Current signal, flat array
        period_length : int
            Compute the active power over period_length samples.

        Returns
        -------
        active_power : ndarray
            Active power signal

        """
        #compute the active power
        metrics = Electrical_Metrics()
        active_power = metrics.active_power(voltage,current,period_length)
        return active_power


    @staticmethod
    def score(ground_truth_events_p, detected_events_p, number_of_samples_in_dataset_p=None,
              tolerance_limit_sec_p=1, return_event_lists_p=False):
        """
        This function can be used to determined the values of the confusion matrix.

        It works as follows:
        For each of the detected events, we build an interval around the detected timestamp using the tolerance_limit_sec
            Then we check for each of the detected intervals if we find a ground truth event that falls within this intervall
             If this is the case we have found a True Positive Event (TP) :
               1. TP += 1
               2. Eliminate this event from the ground truth list, in order to: avoid double detection
               In the case there is more then one ground truth events that fits into the interval, we take the first one
             If this is not the case we have found a False Positive Event (FP):
               1. FP += 1

             After this procedure: all events that are left in the ground truth list have not been detected by the event detection algorithm
             (because they were not removed from the in the previous step)
             Hence those events are actually true, but are delcared to be false by the algorithm: They are False Negatives (FN)
             For each of those events left in the ground truth list: FN += 1

             Finally we determine the number of True Negative Events(TN):
             Lastly, when all the detected and groundtruth events have been processed, the TN are calculated by
             subtracting the TP, FN, and FP from the number of samples in the dataset, i.e., all the positions where an event could have happened
             The start and the end time of the dataset and the information about sampling rate is used to compute the total
             number of possible event timestamps.

        Parameters
        ----------
        ground_truth_events_p : list
            List of datetime objects, containing the ground truth event timestamps.

        detected_events_p : list
            List of datetime objects, containing the event detected by the event detector

        number_of_samples_in_dataset_p : int
            Number of samples, i.e. number of possible event positions in the dataset.
            When computing period wise features, like RMS, values use the corresponding number of samples.
            The value is used to compute the number of True Negative Events.

        tolerance_limit_sec_p : int, optional (default = 1)
            The tolerance limit that is used to determine the scores of the confusion matrix.

        return_events_list_p : boolean, optional (default = False)
            If set to True, the events that are TP, FP, and FN are returned in the results dictionary
        Returns
        -------

        if grid_search_mode is set to True, only the f1score is returned, otherwise:

        results: dictionary
            A dictionary containing the following keys:  tp, fp, fn, tn, f1score, recall, precision, fpp, fpr
            With being:
            tp = True Positives
            fp = False Positives
            fn = False Negatives
            tn = True Negatives
            f1score = F1-Score Metric
            recall = Recall Metric
            precision = Precision Metric
            fpp = False Positive Percentage Metric
            fpr = False Positive Rate Metric

            If return_events_list_p is set to True, the results also contain the following keys:
            tn_events = events that are tn
            fp_events = events that are fp
            fn_events = events that are fn

        """

        ground_truth_events = ground_truth_events_p #ground_truth_events_p.copy() #copy the events
        detected_events = detected_events_p.copy() #copy the events

        results = {}
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        TP_events = []
        FP_events = []
        FN_events = []


        # Compute the Tolerance Intervals around the detected events and iterate through the detected events
        for detected_event in detected_events:
            upper_limit = detected_event + timedelta(seconds=tolerance_limit_sec_p)
            lower_limit = detected_event - timedelta(seconds=tolerance_limit_sec_p)

            # Check which ground_truth events lie within this interval
            matching_ground_truth_events = [(index, ground_truth_event) for index, ground_truth_event in
                                            enumerate(ground_truth_events) if
                                            lower_limit <= ground_truth_event <= upper_limit]

            if len(matching_ground_truth_events) > 0:  # at least (minimum) one ground truth event was detected
                TP += 1
                first_index = matching_ground_truth_events[0][0]

                TP_events.append(ground_truth_events[first_index])
                # It can be possible that there are multiple ground truth events within this interval, hence,
                # we only consider the first one and remove it from the list
                del ground_truth_events[first_index]

            else:  # No Matching Event was found
                FP_events.append(detected_event)
                FP += 1

        # Use the remaining ground truth events to determine the number of FN
        FN = len(ground_truth_events)
        FN_events = ground_truth_events.copy()

        # Use the number of samples in the dataset to determine the TNs in the dataset
        # For BLUED we have data from 20/10/2011 11:58:32.623499 to 27/10/2011 14:10:40.0

        TN = number_of_samples_in_dataset_p - (TP + FP + FN)
        # Compute the final scores
        fpp = FP / (TP + FN)
        fpr = FP / (FP + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1score = (2 * precision * recall) / (precision + recall)

        results["tp"] = TP
        results["fp"] = FP
        results["fn"] = FN
        results["tn"] = TN
        results["f1score"] = f1score
        results["recall"] = recall
        results["precision"] = precision
        results["fpp"] = fpp
        results["fpr"] = fpr

        if return_event_lists_p is True:
            results["tp_events"] = TP_events
            results["fp_events"] = FP_events
            results["fn_events"] = FN_events

        return results

    def _compute_GOF(self,pre_event_window,detection_window):
        """
        When taking the standard chi-squared GOD test formula, one needs to bin the data
        Here it is not necessary as the authors of the paper show

        Therefore: The formula: l_gof = SUM_i_to_n [square((y_i - x_i)) / x_i]

        Parameters
        ----------
        pre_event_window : ndarray
            Pre-event window, flat array
        detection_window : ndarray
            Detection window, flat array


        Returns
        -------
        l_gof : float
            Goodness-of-fit measurement.
        """

        if len(pre_event_window) != len(detection_window):
            raise ValueError("The pre event and the detection window are not equally sized")

        l_gof = []
        for x,y in zip(pre_event_window,detection_window):
            inner_calculation = np.square(y - x) / x #inner part of the formula
            l_gof.append(inner_calculation)

        l_gof = np.sum(np.array(l_gof)) #build the sum over all data samples
        l_gof = np.asscalar(l_gof) #convert the result to a scalar value

        return l_gof

class EventDet_Zheng(BaseEstimator, ClassifierMixin):
    """
   Reference implementation for the following Event Detection algorithm:
        "A Supervised Event-Based Non-Intrusive Load Monitoring for Non-Linear Appliances"

        by: Zheng Zhuang, Hainan Chen, Xiaowei Luo
        link to paper: https://www.researchgate.net/publication/324082872_A_Supervised_Event-Based_Non-Intrusive_Load_Monitoring_for_Non-Linear_Appliances

   The algorithm implementation follows the sklearn API


   Short Description of the algorithm:

        Input: Moving Window Approach --> Pre-Event window

               1. The input is transformed into the active power signal and the current rms
               All values are period wise --> 1/60s temporal resolution
               2. The values are standardized
               2. Suppress minor fluctuations by using Mean Value Filtering and Moving Average Filtering
                ASSUMPTION 1: Mean Value Filtering and Moving Average Filtering essentially is the same
                              Hence, we only implemented one

        DBSCAN Algorithm for clustering the data:
        Two adjacent steady states are considered as two clusters
        Transient intervals are considered as noise or outliers

        Three Hyperparameters: Epsilon (eps) , Mininum Points (min_pts), Window Size (window_size)
            Standard Values from the paper:
            - eps = 0.1
            - min_pts = 25
            - window_size = 300

        Two Main steps:
        1. DBSCAN to detect events in the moving window
                - ASSUMPTION 5:
                There is no information in the paper, about what happens when more then 2 clusters
                are detected. The authors just mention that: "Besides, the window length is determined to make sure
                no more than two clusters are distinguished". Hence, we assume in cases with more than 2
                clusters no event is detected.
                    - A consequence from this ist, that only one event can be detected per window
                - ASSUMPTION 6: The authors do not describe how they exactly determine their event timestamps
                  Hence we assume, that the noisy points that belong to the real transient belong he longest sequence in the
                  indices of the noisy points. Therefore, we determine the longest sequence and we take the central value
                  of this sequence as the exact event point.

        2. Postprocessing with two constraints to eliminate duplicates and to discard unreasonable results
            2.1 |P_pre - P_post| > p_thre
                with P_pre and P_post = active power of pre and post event steady-state
                    - ASSUMPTION 2: authors mention standard value for P_pre, but this is a likely a mistake as
                    only a standard value for p_thre makes sense, hence we set p_thre= 25W
                    - ASSUMPTION 4: We further assume this is the mean value of the datapoints in the pre and post
                    event window
            2.2 if (t(i + 1) - t(i)) < t_thre then discard t(i + 1)
                with t(i+1) and t(i) being to adjacent detect event timestamps,
                t_thre is the minimum time between to successive events
                    - t_thre = 0.01s

            The first postprocessing step 2.1 is performed in the event detector
            The second postprocessing step 2.1 is performed after the event detection is finished:
            it is handled by a additional postprocessing function: postprocess_min_event_distance()
            This functions takes the following input:
                - a pandas dataframe with one column ["Event_Timestamp"]
                - and returns a dataframe with the filtered events
       """

    def __init__(self, network_frequency, eps=0.1, min_pts=25, window_size=300, p_thre=25, t_thre=0.01):
        """
        Initialization Method for the Algorithm Class


        Parameters
        ----------
        network_frequency: int
            Frequency of the power network the data was recorded in (e.g. 50 Hz for Europe)
        eps : float, optional (default=0.1)
            DBSCAN epsilon parameter
        min_pts : int, optional (default=25)
            DBSCAN minimum points paramter
        window_size : int, optional (default=300)
            Window size in samoples
        p_thre : int, optional (default=25)
            Power Threshold
        t_thre : float, optional (default=0.01)
            Time Threshold
        """

        self.eps = eps
        self.min_pts = min_pts
        self.window_size_samples = window_size
        self.p_thre = p_thre
        self.t_thre = t_thre
        self.network_frequency = network_frequency
        self.window_size_sec = self.window_size_samples / self.network_frequency

    def fit(self):
        """
        Function to Fit the Estimator
        As no Training is Required, it just sets the is_fitted parameter to true
        """
        self.is_fitted = True

    def predict(self, active_power, current_rms, start_timestamp, end_timestamp, return_non_postprocessed=True, return_power_thresholds=False):
        """
        Predict if evetns are in the input
        Returns:
            events: list with event timestamp

            optional:
            non_postprocessed_evets: if return_non_postprocessed is True

            optional:
            (mean_pre_average_power, mean_post_average_power): if return_thresholds is True


        Parameters
        ----------
        active_power : ndarray
            Active power signal, flat array
        current_rms : ndarray
            Current RMS values, flat array
        start_timestamp : datetime.datetime
            Start timestamp of the window
        end_timestamp : datetime.datetime
            End timestamp of the window


        Returns
        -------
        events : list
            List with event timestamps in the window
        non_postprocessed_events : list, optional
            If return_non_postprocessed is True, the non post-processed events are returned
        (mean_pre_average_power, mean_post_average_power) : tuple, optional
            If return_thresholds is True, the thresholds computed are returned

        """


        check_is_fitted(self, ["is_fitted"]) #check if fit() was called before

        #Reshape the input data into one 2-dimensional input
        if len(active_power) != len(current_rms):
            raise ValueError("The active power and the current rms input has to have the same length")

        input_data = np.stack((active_power, current_rms),axis=-1) #new 2 dimensional input
        assert input_data.shape == (len(active_power),2)

        # Standardize the input
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)

        # Do the clustering

        # get the input data again
        active_power = input_data[:, 0]
        current_rms = input_data[:, 1]



        dbscan = DBSCAN(eps=self.eps,min_samples=self.min_pts).fit(input_data)


        dbscan_labels = np.array(dbscan.labels_)
        #Noise samples get the "-1" class --> those are the transients
        #In order to detect an event we need 2 steady states and noise --> 3 cluster
        cluster_labels = np.unique(dbscan_labels)
        number_of_clusters = len(cluster_labels) #get the number of clusters
        mean_pre_steady_state_power = 0
        mean_post_steady_state_power = 0
        events = []
        non_post_processed_events = []
        if number_of_clusters == 3 and len(np.where(dbscan_labels==-1)) > 0 : #for events the number_of_clusters should be 3
            #the second conditions checks if there are -1 = noisy data points = transients in the signal
            event_data_points_indices = np.where(dbscan_labels==-1)

            #the first cluster gets the index 0, the second index 1 - always

            #Select the indices where the labels are 0

            # ASSUMPTION 6: detect the exact event point by finding the longest sequence
            start_index_of_longest_sequence, length_of_longest_sequence = self._return_longest_sequence(event_data_points_indices)
            longest_sequence = event_data_points_indices[start_index_of_longest_sequence:int(start_index_of_longest_sequence + length_of_longest_sequence)]
            event_index = math.ceil(np.median(np.array(longest_sequence)))

            event_seconds = event_index * (1 / self.network_frequency)
            # compute the corresponding event timestamp
            event_timestamp = start_timestamp + timedelta(seconds=event_seconds)
            non_post_processed_events = [event_timestamp]

            pre_steady_state_power = active_power[:event_index]
            post_steady_state_power = active_power[event_index:]

            mean_pre_steady_state_power = np.mean(pre_steady_state_power)
            mean_post_steady_state_power = np.mean(post_steady_state_power)

            #Postprocessing step 2.1 - Filter the step size
            if not np.abs(mean_pre_steady_state_power - mean_post_steady_state_power) > self.p_thre:
                events = []

            else: #event is detected
                events = [event_timestamp]

        else: #no Event Detected
            events = []
            non_post_processed_events = []

        if return_non_postprocessed == True:

            if return_power_thresholds == True:

                return events, non_post_processed_events, mean_pre_steady_state_power, mean_post_steady_state_power
            else:
                return events, non_post_processed_events,
        else:
            if return_power_thresholds == True:

                return events, mean_pre_steady_state_power, mean_post_steady_state_power
            else:
                return events

    def compute_input_signal(self, voltage, current, period_length):
        """
        1. The input is transformed into the active power signal and the current rms
           All values are period wise --> 1/60s temporal resolution
           1. The values are standardized
           2. Suppress minor fluctuations by using Mean Value Filtering and Moving Average Filtering, what is essentially the same
            ASSUMPTION 3: Rolling Window for averaging of size 3

        Parameters
        ----------
        voltage : ndarray
            Voltage signal, flat array
        current : ndarray
            Current signal, flat array
        period_length : int
            Number of samples the features are computed over.

        Returns
        -------
        active_power : ndarray
            Active power signal
        current_rms : ndarray
            Root-Mean-Square values of the current

        """
        # 1. Compute the Metrics
        Metrics = Electrical_Metrics()
        current_rms = Metrics.compute_single_rms(signal=current, period_length=period_length, )
        active_power = Metrics.active_power(instant_voltage=voltage, instant_current=current,
                                            period_length=period_length)

        # 2. Smoothening
        # 2.1 Mean Value Filtering/Smoothening
        current_rms = pd.Series(current_rms)
        active_power = pd.Series(active_power)

        current_rms_rolling = current_rms.rolling(window=2)  # rolling windows of size 2
        current_rms = np.array(current_rms_rolling.mean())
        current_rms = current_rms[~np.isnan(current_rms)]  # remove nan values
        active_power_rolling = active_power.rolling(window=2)  # rolling windows of size 2
        active_power = np.array(active_power_rolling.mean())
        active_power = active_power[~np.isnan(active_power)]  # remove nan values

        return active_power, current_rms

    @staticmethod
    def score(ground_truth_events_p, detected_events_p, number_of_samples_in_dataset_p=None,
              tolerance_limit_sec_p=1, return_event_lists_p=False):
        """
        This function can be used to determined the values of the confusion matrix.

        It works as follows:
        For each of the detected events, we build an interval around the detected timestamp using the tolerance_limit_sec
            Then we check for each of the detected intervals if we find a ground truth event that falls within this intervall
             If this is the case we have found a True Positive Event (TP) :
               1. TP += 1
               2. Eliminate this event from the ground truth list, in order to: avoid double detection
               In the case there is more then one ground truth events that fits into the interval, we take the first one
             If this is not the case we have found a False Positive Event (FP):
               1. FP += 1

             After this procedure: all events that are left in the ground truth list have not been detected by the event detection algorithm
             (because they were not removed from the in the previous step)
             Hence those events are actually true, but are delcared to be false by the algorithm: They are False Negatives (FN)
             For each of those events left in the ground truth list: FN += 1

             Finally we determine the number of True Negative Events(TN):
             Lastly, when all the detected and groundtruth events have been processed, the TN are calculated by
             subtracting the TP, FN, and FP from the number of samples in the dataset, i.e., all the positions where an event could have happened
             The start and the end time of the dataset and the information about sampling rate is used to compute the total
             number of possible event timestamps.

        Parameters
        ----------
        ground_truth_events_p : list
            List of datetime objects, containing the ground truth event timestamps.

        detected_events_p : list
            List of datetime objects, containing the event detected by the event detector

        number_of_samples_in_dataset_p : int
            Number of samples, i.e. number of possible event positions in the dataset.
            When computing period wise features, like RMS, values use the corresponding number of samples.
            The value is used to compute the number of True Negative Events.

        tolerance_limit_sec_p : int, optional (default = 1)
            The tolerance limit that is used to determine the scores of the confusion matrix.

        return_events_list_p : boolean, optional (default = False)
            If set to True, the events that are TP, FP, and FN are returned in the results dictionary
        Returns
        -------

        If grid_search_mode is set to True, only the f1score is returned, otherwise:

        results: dictionary
            A dictionary containing the following keys:  tp, fp, fn, tn, f1score, recall, precision, fpp, fpr
            With being:
            tp = True Positives
            fp = False Positives
            fn = False Negatives
            tn = True Negatives
            f1score = F1-Score Metric
            recall = Recall Metric
            precision = Precision Metric
            fpp = False Positive Percentage Metric
            fpr = False Positive Rate Metric

            If return_events_list_p is set to True, the results also contain the following keys:
            tn_events = events that are tn
            fp_events = events that are fp
            fn_events = events that are fn

        """

        ground_truth_events = ground_truth_events_p #ground_truth_events_p.copy() #copy the events
        detected_events = detected_events_p.copy() #copy the events

        results = {}
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        TP_events = []
        FP_events = []
        FN_events = []


        # Compute the Tolerance Intervals around the detected events and iterate through the detected events
        for detected_event in detected_events:
            upper_limit = detected_event + timedelta(seconds=tolerance_limit_sec_p)
            lower_limit = detected_event - timedelta(seconds=tolerance_limit_sec_p)

            # Check which ground_truth events lie within this interval
            matching_ground_truth_events = [(index, ground_truth_event) for index, ground_truth_event in
                                            enumerate(ground_truth_events) if
                                            lower_limit <= ground_truth_event <= upper_limit]

            if len(matching_ground_truth_events) > 0:  # at least (minimum) one ground truth event was detected
                TP += 1
                first_index = matching_ground_truth_events[0][0]

                TP_events.append(ground_truth_events[first_index])
                # It can be possible that there are multiple ground truth events within this interval, hence,
                # we only consider the first one and remove it from the list
                del ground_truth_events[first_index]

            else:  # No Matching Event was found
                FP_events.append(detected_event)
                FP += 1

        # Use the remaining ground truth events to determine the number of FN
        FN = len(ground_truth_events)
        FN_events = ground_truth_events.copy()

        # Use the number of samples in the dataset to determine the TNs in the dataset
        # For BLUED we have data from 20/10/2011 11:58:32.623499 to 27/10/2011 14:10:40.0

        TN = number_of_samples_in_dataset_p - (TP + FP + FN)
        # Compute the final scores
        fpp = FP / (TP + FN)
        fpr = FP / (FP + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1score = (2 * precision * recall) / (precision + recall)

        results["tp"] = TP
        results["fp"] = FP
        results["fn"] = FN
        results["tn"] = TN
        results["f1score"] = f1score
        results["recall"] = recall
        results["precision"] = precision
        results["fpp"] = fpp
        results["fpr"] = fpr

        if return_event_lists_p is True:
            results["tp_events"] = TP_events
            results["fp_events"] = FP_events
            results["fn_events"] = FN_events

        return results

    def _return_longest_sequence(self, sequence):
        """
        Return the longest sequence in the signal

        Parameters
        ----------
        sequence : ndarray
            Sequence of detections

        Returns
        -------
        start_index_of_longest_sequence : int
            Start index of the longest sequence detected
        length_of_longest_sequence : int
            Length of the longest sequence detected

        """
        if len(sequence) == 0:
            raise ValueError("Can not find the longest sequence in an empty sequence")

        if len(sequence) == 1:  # single value sequence
            return 0, 1  # index, length_of_longest_sequence

        start_index_of_longest_sequence = None
        length_of_longest_sequence = 1  # a single element "sequence" has the length 1

        start_of_current_sequence = None
        length_of_current_sequence = 1

        for index, element in enumerate(sequence):

            if index != len(sequence) - 1:  # for all elements except the last one
                if element + 1 == sequence[index + 1]:  # if we have a sequence
                    if start_of_current_sequence is None:  # if the first element in a new sequence
                        start_of_current_sequence = index
                        length_of_current_sequence += 1
                    else:  # not the first index in the curren sequence
                        length_of_current_sequence += 1

                    # Compare the Sequence to the longest one
                    if length_of_longest_sequence < length_of_current_sequence:
                        length_of_longest_sequence = length_of_current_sequence
                        start_index_of_longest_sequence = start_of_current_sequence

                else:  # if no sequence detected or if sequence has stopped
                    start_of_current_sequence = None
                    length_of_current_sequence = 1

        return start_index_of_longest_sequence, length_of_longest_sequence

    def postprocess_min_event_distance(self, event_list):
        """
        Postprocessing step 2.2
        This function is called after the event detection has finished
        It ensure a minimum distance between consecutive events.
        If this distance (in time) is below a preset threshold one event is dropped
        Hence,  if (t(i + 1) - t(i)) < t_thre then discard t(i + 1)

        Parameters
        ----------
        event_list : list
            List of datetime.datetime event objects

        Returns
        -------
        postprocessed_events : list
            List of datetime.datetime event objects that have been post-processed

        """

        postprocessed_events = []

        event_already_appended = False  # flag to make sure no event is added two times
        for index, event_ts in enumerate(event_list):
            time_difference = event_list[int(index + 1)] - event_ts
            if time_difference < self.t_thre:  # drop the t+1 event
                if event_already_appended is False:
                    postprocessed_events.append(event_ts)  # only append event t, not t+1 --> this event is dropped
                event_already_appended = False  # the next event is not appended
            else:
                if event_already_appended is False:  # check if the current event was already appended in the previous step
                    postprocessed_events.append(event_ts)
                postprocessed_events.append(event_list[int(index + 1)])  # also append the next one
                # now we need to make sure that this event is not appended in the next case
                event_already_appended = True

        return postprocessed_events

class EventDet_Barsim_Sequential(BaseEstimator, ClassifierMixin):
    """
    Reference implementation for the following Event Detection algorithm:
        "sequential clustering-based event detection for nonintrusive load monitoring"

        by: Karim Said Barsim and Bin Yang
        link to paper: https://pdfs.semanticscholar.org/74db/87eb3e17a2af1c4e411e2c0677ac0d20f9dc.pdf

    The algorithm implementation follows the general sklearn API.

    Use the algorithm and this class as follows:
    - create an instance of the algorithm class
    - call the fit() method
    - prepare the input window with the raw current and voltage values according to the description below
    - call the compute_input_signal() method to compute the features the algorithm requires.

    ATTENTION: It is necessary to use the compute_input_signal() function, because the raw values handed
    to the function are further used to ensure to correct input order of the input windows and to check
    if the relative offsets are in line with the ones returned from the previous window.
    The return values and offsets of the predict() function are further explained below.

    - call the predict() method on the features to detect events in the window
    - proceed with streaming the next window, depending on the result returned by the predict() method, as
    described below.

    Hence, there are three essential external methods available that are the central API of the algorithm:
    fit(), compute_input_signal() and predict().
    The fourth method _convert_relative_offset can be used to convert the offsets that are returned
    by the predict() method which are related to the input data, as computed by the compute_input_signal() function
    back to be relative to the raw input data, what is useful for the streaming process.


    Short Description of the algorithm and the input it requires:

        Input:
            real (P) and reactive powers (Q) at a time instant, approximately computed periodewise
            and averaged over periods.
            We compute 2 values per second.
            Every sample point at time t therefore has two measurements [P,Q]

            The input that is expected by the event detector can be obtained by calling the
            compute_input_signal method of the event detector.

            Due to the inner workings of the algorithm, the input that is needed is longer then the length of
            the initial input window X (with window_samples_n datapoints in it). Therefore, the input X_overall
            that is fed to the algorithm, will be split up into two arrays, namely the initial input window X.
            and the remaining datapoints after X, namely the X_future.
            We recommend feeding at least as many window_samples_n datapoints that occur after the input window X to the
            algorithm, i.e. the X_overall input should have a length of at least 2 times window_samples_n.

            The future_window_size_n parameter that is set during initialization determines the size of the
            X_future array.

            As you see in the description below, the datapoints from the X_future array are appended one by one
            two the input window X.

            ATTENTION:
            At the end, there are two cases, i.e. either an event is detected or not:
            If an event is detected, two indices are returned, the beginning and the end of the event-interval
            that was detected, else None is returned. The next fixed input window X that you should feed to the
            algorithm, should start at the end of the event-interval (i.e. the second index + 1 that is returned).
            So there is some overlap between the windows!
            If no event is detected, you should continue with the next window. Most of the data contains no event,
            hence, to speed up the algorithm, no overlap is created when no event was detected.



        The DBSCAN Algorithm is used for clustering the input data:
            It has three Hyperparameters: Epsilon (eps) , Mininum Points (min_pts), Window Size (window_size)
            The parameter values are not mentioned in the paper, hence we did an extensive grid search.
            With a focus on precision, instead of recall we recommend using.
            - eps = 0.05
            - min_pts = 3
            - window_size = 5


        In their paper, the authors define three event models, each of them oopses constraints a detected
        event has to fullfill. In this implementation, we use the same model the authors have used in their
        benchmark, i.e. Model 3. Model 3 is the most general one of the models, i.e. model 1 and model 2
        are specialisations with more restrictions of event-model 3.


        Event-model 3 is specified as follows:
        For the definition and the algorithm we need to define two distinct points of each cluster, that are
        important to compute multiple metrics.
            - u: is the index of the first sample (with respect to time) n the cluster Ci (all other points that are not in the
            cluster Ci have to have a smaller index thant u)
            - v: is the index of the last sample (with respect to time) in the cluster Ci

        A sequence of samples X is defined as an event if:
            (1) it contains at least two clusters C1 and C2 (besides the outlier cluster) and the outlier Cluster
            C0 can be non empty.

            (2) clusters C1 and C2 have a high temporal locality, i.e. Loc(Ci) >= 1 - temp_eps
            with Loc(Ci) = n_samples_in_Ci / (v - u + 1)
            i.e. there are at least two, non noise, clusters with a high temporal locality

            (3) clusters C1 and C2 do not interleave in the time domain.
            There is a point s in C1 for which all points n > s do not belong to C1 anymore, i.e. s is the upper
            bound of C1 in this case.
            There is also a point i in C2 for which all points n < i do not belong to C2 anymore, i.e. i is
            the lower bound of C2 in this case.
            (Note: changed naming here, to avoid confusion with the definition of u and v above, although
            the meaning is the same)


        In order to fulfill these requirements we need to find at least two clusters that pass the checks.
        More then two clusters are fine. Checks (1) and (2) are performed independently, check (3)
        is performed for the remaining clusters that fullfill the pass the checks (1) and (2)

        The model constraints are implemented in the _check_event_model_constraints() method.
        If one intends to use another event model, this method has to be overwritten accordingly.

        By identifying two clusters that fulfill these requirements, we likely have discovered two
        stationary segments, consisting of sample belonging to C1 and C2. In between we have the change interval
        we are looking for, i.e. the event interval. The points in between the intervals are somehow considererd
        to be the event, if one looks closely at the samples in Figure 1 of the paper, especially in subfigure c.
        Hence all points that are in between the upper bound of C1 and the lower bound of C2, that are
        within the noise cluster (See p. 80 of the Paper). The transient is noise, that is detected by the DBSCAN.

        (Note: the upper bound of C1 < lower bound of C2, using this we can decide which cluster we name C1)
        We name points in between the event_interval_t (X_t in the paper).

        The algorithm can be divided in two steps, namely, a forward and a backward pass.
        The forward pass is used to find events, i.e. change-point intervals, and the backward pass is used
        to improve the segmentation of the input signal into steady-state and event intervals.

        1. Forward pass:
                For a given input vector X with length n o the following:
                    1. Take the next sample x_n+1 and append it to X
                    2. Update the clustering and the clustering structure, using the DBSCAN Algorithm
                    By doing this we get clusters C1 and C2 and a possible event_interval_t
                    3. Compute the loss for the given cluster and the given event model (i.e. Model 3)
                       The loss for a signal segment X and two clusters C1 and C2 is defined as follows,
                       it counts the number of samples that need to be corrected in order for the segment
                       to match the event model:
                       L(X) = a + b + c with
                       a: number of samples n from C2 with n <= u, with u being the lower bound of
                       the event_interval_t
                       b: number of samples n from C1 with n >= v, with v being the upper bound of
                       the  event_interval_t
                       c: number of samples n between u < n < v, so number of samples n in the event_interval_t that
                       belong to C1 or C2, i.e. to the stationary signal.

                       We compute this loss for all cluster combinations, i.e. if the event model checks are
                       passed by three (non noise) clusters, then we compute two loss values

                    4. Check if L(X) <= loss_thresh.
                    If not go to step 1. and take the next sample.

                    Note: we have included a savety mechanism to prevent memory errors: if the size of X is bigger
                    then future_window_size_n times of the original window size, then we return that
                    no event was detected and the user should continue with the next input as described in the
                    input section of the documentation.

                    If yes:
                       if multiple cluster combinations have passed the loss_thresh, then declare
                       an event as detected  detected, with the change-interval event_interval_t
                       that results from the cluster combination with the lowest loss L(X)
                       Go to step 5 and start the backward pass.

        2. Backward pass:
                    1. Delete the oldest sample x1 from the segment (i.e the first sample in X)
                    2. Update the clustering and the clustering structure, using the DBSCAN Algorithm
                    3. Check the loss L(X) for the detected segment and the detected event_interval_t.
                    If L(X) <= loss_thresh, go to step 2 again.
                    If L(X=) >= loss_thresh, i.e. if without the removed sample no event is detected anymore
                    reinsert the last-sample again and declare the segment X as a balanced event

        After this is done, the process restarts from the first sample of C2 (x_v).

        The whole algorithm is window-based, with a initial window size of window_size_n
        The event detector has the following hyperparameters that can be fine-tuned.
            - DBSCAN Epsilon (dbscan_eps)
            - DBSCAN Mininum Points (dbscan_min_pts)
            - Window Size (window_size_n)
            - Threshold for the Loss-Function (loss_thresh)
            - Temporal Locality Epsilon (temp_eps)

        Non algorithm related parameters that can be adapted are:
            - Number of datapoints after the input window that are fed to the algorithm future_window_size_n
            - Flag to turn input window order checks on or of perform_input_order_checks
            (see also the details in the input section of this documentation for the two parameters above)

        If you want to debug the inner workings of the algorithm, i.e. get plots on the clustering etc. .
        then initialize the estimator with debugging_mode=False.
        Using this in a graphical environment, like a jupyter notebook is highly recommended.
        This provides a tool to understand the inner workings of the algorithm in detail.
   """

    def __init__(self, dbscan_eps=0.05, dbscan_min_pts=3, window_size_n=5, future_window_size_n=5,
                 loss_thresh=40, temp_eps=0.8, debugging_mode=False, dbscan_multiprocessing=False, network_frequency=60, **kwargs):
        """

         Args:
            dbscan_eps (float): Epsilon Parameter for the DBSCAN algorithm
            dbscan_min_pts (int): Minimum Points Parameter for the DBSCAN algorithm
            window_size_n (int):
            future_window_size_n (int):

            loss_thresh (int):
            temp_eps (float):  t
            perform_input_order_checks:
            debugging_mode (bool): activate if plots of the dbscan clustering shall be shown
            grid_search_mode: activate to adapt the score function, if you want to perfrom grid-search
            dbscan_multiprocessing (bool): default=False, if set to true multiple processes are used in the dbscan algorithm.
            If the Barsim_Sequential event detector is used within a multiprocessing environment, turning the dbscan_multiprocessing
            paramter to True, results in warnings by sklearn and the multiprocessing library, as no additional subprocesses can
            be spawned by the processes.

        Parameters
        ----------
        dbscan_eps : float, optional (default=0.05)
            Epsilon Parameter for the DBSCAN algorithm
        dbscan_min_pts : int, optional (default=3)
            Minimum Points Parameter for the DBSCAN algorithm
        window_size_n : int, optional (default=5)
            Window Size
        future_window_size_n : int, optional (default=5)
            Maximum Number of samples that are gradually added to the window_size_n window.
            window_size_n +n future_window_size_n is the maximum possible window, then no event detected is returned
            in case.
        loss_thresh : int, optional (default=40)
            Threshold for the loss-function
        temp_eps : float, optional (default=0.8)
            Temporal locality epsilon
        debugging_mode : boolean, optional (default=False)
            Activate if plots of the dbscan clustering shall be shown
        dbscan_multiprocessing : boolean, optional (default=False)
            default=False, if set to true multiple processes are used in the dbscan algorithm.
            If the Barsim_Sequential event detector is used within a multiprocessing environment, turning the dbscan_multiprocessing
            parameter to True, results in warnings by sklearn and the multiprocessing library, as no additional subprocesses can
            be spawned by the processes.
        network_frequency : int, optional (default=60)
            Frequency of the power network the data was recorded at.
        kwargs** : other keyword arguments
        """


        self.dbscan_eps = dbscan_eps
        self.dbscan_min_pts = dbscan_min_pts
        self.window_size_n = window_size_n
        self.future_window_size_n = future_window_size_n
        self.loss_thresh = loss_thresh
        self.temp_eps = temp_eps

        self.network_frequency = network_frequency # periods per second

        # initialize the corresponding parameter for the checks
        self.order_safety_check = None



        self.debugging_mode = debugging_mode

        self.dbscan_multiprocessing=dbscan_multiprocessing

        # Display some warning if the window_size_n and the future_window_size_n is unbalanced
        if future_window_size_n / window_size_n < 5:
            warnings.warn("You have provided less then 5 times of future_window_samples_n than window_size_n samples "
                          "to the algorithm. Please make sure that your window_size_n is at least big enough for the "
                          "event detector to work. We recommend using more future samples", Warning)

    def fit(self):
        """
        Call before calling the predict function. Needed by the sklearn API.

        """

        self.is_fitted = True

    def predict(self, X_overall):
        """
        Predict if the input provided contains an event or not.
        The input provided should be computed by the compute_input_signal function of this class.

        Parameters
        ----------
        X_overall:  ndarray
            Input computed by compute_input_signal function.

        Returns
        -------
        event_interval_indices : tuple
            (start_index, end_index), if no event detected None is returned

        """
        # Check if fit was called before
        check_is_fitted(self, ["is_fitted"])

        # 1. Check the input
        # 1.1 Perform general tests on the input array

        utils.assert_all_finite(X_overall)
        X_overall = utils.as_float_array(X_overall)

        # 1.2 Check the size of X_overall
        if not len(X_overall) == self.window_size_n + self.future_window_size_n:
            raise ValueError("The input size of X_overall (" + str(len(X_overall)) + ") does not match the"
             "window_size_n (" + str(self.window_size_n) + ") + the future_window_samples_n (" +
                               str(self.future_window_size_n) + ") parameters")


        # 1.3 Split the input array into the initial inout window X and the remaining X_future datapoints
        X, X_future = np.split(X_overall, [self.window_size_n])


        # 2. Event Detection Logic

        # 2.1 Forward Pass

        # We need to loop over all samples in X_future, until an event is detected
        # In case no event is detected and all samples in X_future are added to X, we return None

        # 2.1.1 Take the next sample x_n+1 and append it to X
        if self.debugging_mode == True:
            processing_start_time = time.process_time()

            self.datapoints_added = 0

        for new_datapoint in X_future:
            if self.debugging_mode == True:
                self.datapoints_added += 1

            event_detected = False # Flag to indicate if an event was detected or not

            X = np.concatenate([X, [new_datapoint]])

            # 2.1.2 Update the clustering and the clustering structure, using the DBSCAN Algorithm
            # By doing this we get clusters C1 and C2
            self._update_clustering(X)

            # Now check the mode constraints
            # Possible intervals event_interval_t are computed in the _check_event_model_constraints() function.
            checked_clusters = self._check_event_model_constraints()

            # If there are no clusters that pass the model constraint tests, the method _check_event_model_constraints()
            # returns None, else a list of triples (c1, c2, event_interval_t).

            if checked_clusters is None:
                continue #add the next new_datapoint. We go back to step 1 of the forward pass.

            # 2.1.3 Compute the Loss-values

            else: # at least one possible combination of two clusters fullfills the event model constraints-
                # Hence, we can proceed with step 3 of the forward pass.
                # Therefore, we compute the loss for the given cluster combination.
                # The formula for the loss, is explained in the doc-string in step 3 of the forward pass.

                event_cluster_combination = self._compute_and_evaluate_loss(checked_clusters)
                self.forward_clustering_structure = self.clustering_structure #save the forward clustering structure

                if event_cluster_combination is not None: #  event detected
                    event_detected = True
                    break #leave the loop of adding new samples each round and continue the code after the loop

                else: # go back to step 1 and add the next sample
                    continue


        if event_detected == True: #an event was detected in the forward pass, so the backward pass is started
            if self.debugging_mode == True:
                print("Event Detected at: " + str(event_cluster_combination))
                print("")
                print("")
                print(60*"*")
                print("Backward pass is starting")
                print(60 * "*")

            # Initialize the backward pass clustering with the forward pass clustering, in case already the
            # first sample that is removed, causes the algorithm to fail. Then the result from the forward
            # pass is the most balanced event
            self.backward_clustering_structure = self.forward_clustering_structure
            event_cluster_combination_balanced = event_cluster_combination

            # 2.2.1. Delete the oldest sample x1 from the segment (i.e the first sample in X)
            for i in range(1, len(X)-1):
                X_cut = X[i:] #delete the first i elements, i.e. in each round the oldest sample is removed

                # 2.2.2 Update the clustering structure
                self._update_clustering(X_cut) #the clustering_structure is overwritten, but the winning one
                # from the forward pass is still saved in the forward_clustering_structure attribute

                # 2.2.3 Compute the loss-for all clusters that are detected (except the detected)
                # Hence, we need to check the event model constraints again
                checked_clusters = self._check_event_model_constraints()

                if checked_clusters is None: #roleback with break
                    status = "break"
                    event_cluster_combination_balanced = self._roleback_backward_pass(status, event_cluster_combination_balanced,i)
                    break #finished

                else: #compute the loss
                    # 2.2.4 Check the loss-values for the detected segment
                    event_cluster_combination_below_loss = self._compute_and_evaluate_loss(checked_clusters)

                    if event_cluster_combination_below_loss is None: #roleback with break
                        status = "break"
                        event_cluster_combination_balanced = self._roleback_backward_pass(status,
                                                                                          event_cluster_combination_balanced,
                                                                                          i)
                        break #finished
                    else:
                        status = "continue"
                        event_cluster_combination_balanced = self._roleback_backward_pass(status,
                                                                                          event_cluster_combination_balanced,
                                                                                          i,
                                                                                          event_cluster_combination_below_loss
                                                                                          )
                        continue #not finished, next round, fiight

            event_start = event_cluster_combination_balanced[2][0]
            event_end = event_cluster_combination_balanced[2][-1]
            if self.debugging_mode == True:
                print("Balanced event detected in the Backward pass from " + str(event_start) + " to " + str(event_end))
            # In case an event is detected, the first sample of the second steady state segment (c2) should be fed to
            # the estimator again for further event detection, as described in the documentation.
            # We use the first 10 start values to perform the necessary input check if the corresponding parameter
            # perform_input_order_checks = True


            self.order_safety_check = {"first_10_start_values" : X[event_end + 1: event_end + 11] }

            if self.debugging_mode is True:
                elapsed_time = time.process_time() - processing_start_time
                print("")
                print("Processing this window took: " + str(elapsed_time) + " seconds")



            return (event_start, event_end)

        else:
            # also for the input order check
            # in case no event is detected, the user should feed back the last window_size_n samples of X.
            # this is implemented that way to prevent memory issues
            self.order_safety_check = {"first_10_start_values": X[-self.window_size_n:][:10]}

            if self.debugging_mode is True:
                elapsed_time = time.process_time() - processing_start_time
                print("")
                print("Processing this window took: " + str(elapsed_time) + " seconds")

            return None

    def _compute_and_evaluate_loss(self, checked_clusters):
        """
        Function to compute the loss values of the different cluster combinations.
        The formula for the loss, is explained in the doc-string in step 3 of the forward pass.

        Parameters
        ----------
        checked_clusters: list
            Clusters checked by step 2

        Returns
        -------
        event_cluster_combination: dictionary
            Event cluster combinations that have passed the step 3 check

        """

        if self.debugging_mode is True:
            print("")
            print("")
            print("Compute the Loss values for all cluster combinations that have passed the model constraints")
            print("They have to be smaller than: " + str(self.loss_thresh))

        event_model_loss_list = []
        for c1, c2, event_interval_t in checked_clusters:
            lower_event_bound_u = event_interval_t[0] - 1  # the interval starts at u + 1
            upper_event_bound_v = event_interval_t[-1] + 1  # the interval ends at v -1
            c1_indices = self.clustering_structure[c1]["Member_Indices"]
            c2_indices = self.clustering_structure[c2]["Member_Indices"]
            a = len(np.where(c2_indices <= lower_event_bound_u)[0]) # number of samples from c2 smaller than lower bound of event

            b = len(np.where(c1_indices >= upper_event_bound_v)[0]) # number of samples from c1 greater than upper bound of event

            c1_and_c2_indices = np.concatenate([c1_indices, c2_indices])

            # number of samples n between u < n < v, so number of samples n in the event_interval_t that
            # belong to C1 or C2, i.e. to the stationary signal.
            c = len(np.where((c1_and_c2_indices > lower_event_bound_u) & (c1_and_c2_indices < upper_event_bound_v))[0])


            event_model_loss = a + b + c

            event_model_loss_list.append(event_model_loss)

            if self.debugging_mode is True:
                print("\tLoss for clusters " + str(c1) + " and " + str(c2) + ": " + str(event_model_loss))
                print("\t\tComposited of: " + "a=" + str(a) + " b=" + str(b) + " c=" +str(c))

        # 2.1.4 Compare loss value to the threshold on the loss loss_thresh
        # We select the cluster combination with the smallest loss, that is below the threshold

        # Select the smallest loss value
        min_loss_idx = np.argmin(event_model_loss_list)  # delivers the index of the element with the smallest loss

        # Compare with the loss threshold, i.e. if the smallest loss is not smaller than the treshold, no other
        # loss will be in the array

        if event_model_loss_list[min_loss_idx] <= self.loss_thresh:  # if smaller than the threshold event detected
            event_cluster_combination = checked_clusters[min_loss_idx]  # get the winning event cluster combination
            if self.debugging_mode is True:
                print("\tEvent Cluster Combination determined")
            return event_cluster_combination

        else:
            return None

    def _update_clustering(self, X):
        """
        Using the DBSCAN Algorithm to update the clustering structure.
        All available CPUs are used to do so.
        Furthermore all relevant metrics are directly computed from the clustering result.

        The method sets the clustering_structure attribute of the estimator class:
            clustering_structure (dict): resulting nested clustering structure. contains the following keys
            For each cluster it contains: {"Cluster_Number" : {"Member_Indices": []"u" : int,"v" : int,"Loc" : float} }
            u and v are the smallest and biggest index of each cluster_i respectively.
            Loc is the temporal locality metric of each cluster_i.


        Parameters
        ----------
        X : ndarray
            Input

        """



        # Do the clustering
        # Use all CPU's for this, i.e. set n_jobs = -1
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_pts, n_jobs=-1).fit(X)

        # Get the cluster labels for each datapoint in X
        X_cluster_labels = np.array(dbscan.labels_)

        # Noise samples get the "-1" class --> those are usually the transients
        # Get all unique cluster identifiers
        cluster_labels = np.unique(X_cluster_labels)


        if self.debugging_mode == True: #if in debuggin mode, plot the clusters

            if self.original_non_log is False:
                log_label = "(Log-Scale)"
            else:
                log_label = ""

            plt.clf()
            plt.scatter(x=np.arange(len(X)),y=X[:, 0], c=X_cluster_labels, cmap='Paired')
            plt.ylabel("Active Power " + log_label)
            plt.xlabel("Samples")
            plt.title("Clustering")
            plt.show()
            plt.clf()
            plt.scatter(x=X[:, 1], y=X[:,0],  c=X_cluster_labels, cmap='Paired')
            plt.ylabel("Active Power " + log_label)
            plt.xlabel("Reactive Power " + log_label)
            plt.title("Clustering")
            plt.show()

        clustering_structure = {}

        #build the cluster structure, for each cluster store the indices of the points.
        for cluster_i in cluster_labels:
            cluster_i_structure = {} #all the relvant information about cluster_i

            # Which datapoints (indices) belong to cluster_i
            cluster_i_member_indices = np.where(X_cluster_labels == cluster_i)[0]
            cluster_i_structure["Member_Indices"] = np.array(cluster_i_member_indices)

            # Determine u and v of the cluster (the timely first and last element, i.e. the min and max index)
            u = np.min(cluster_i_member_indices)
            v = np.max(cluster_i_member_indices)
            cluster_i_structure["u"] = u
            cluster_i_structure["v"] = v

            # compute the temporal locality of cluster_ci
            Loc_cluster_i = len(cluster_i_member_indices) / (v - u + 1) # len(cluster_i_member_indices) = n_samples_in_Ci
            cluster_i_structure["Loc"] = Loc_cluster_i

            # insert the structure of cluster_i into the overall clustering_structure
            clustering_structure[cluster_i] = cluster_i_structure


        self.clustering_structure = clustering_structure

        return None

    def _roleback_backward_pass(self, status, event_cluster_combination_balanced, i, event_cluster_combination_below_loss=None):
        """
        When the backward pass is performed, the oldest datapoint is removed in each iteration.
        After that, first the model constraints are evaluated.
        If they are violated, we roleback to the previous version by adding the oldest datapoint again
        and we are finished.
        In case the model constraints still hold, we recompute the loss.
        If the loss exceeds the threshold, we ne to roleback to the last version too.

        This roleback happens at to positions in the code (i.e. after the model constraints are evaluated and after
        the loss computation). Therefore, it is encapsulated in this function.


        Parameters
        ----------
        status : str
            Status after the backward pass check.
            Either "continue" or "break"
        event_cluster_combination_balanced : dictionary
            Balanced event cluster combinations
        i : int
            current iteration index of the datapoint
        event_cluster_combination_below_loss : dictionary
            Event clusters combinations smaller then the loss value

        Returns
        -------

        """
        if status == "break":
            # if the loss is above the threshold
            # without the recently removed sample, take the previous combination and declare it as an
            # balanced event.
            # the previous clustering and the previous event_cluster_combination are saved from the previous
            # run automatically, so there is no need to perform the clustering again.

            # Attention: the event_interval indices are now matched to X_cut.
            # We want them to match the original input X instead.
            # Therefore we need to add + (i-1) to the indices, the  -1 is done because we take
            # the clustering and the state of X_cut from the previous, i.e. i-1, round.
            # This is the last round where the loss, was below the threshold, so it is still fine
            event_cluster_combination_balanced = list(event_cluster_combination_balanced)
            event_cluster_combination_balanced[2] = event_cluster_combination_balanced[2] + i  # the event_interval_t
            event_cluster_combination_balanced = tuple(event_cluster_combination_balanced)


            # The same is to be done for all the final cluster
            # The structure stored in self.backward_clustering_structure is valid, it is from the previous iteration
            for cluster_i, cluster_i_structure in self.backward_clustering_structure.items():
                cluster_i_structure["Member_Indices"] = cluster_i_structure["Member_Indices"] + int(i - 1)
                cluster_i_structure["u"] = cluster_i_structure["u"] + int(i - 1)
                cluster_i_structure["v"] = cluster_i_structure["v"] + int(i - 1)

                # Only the "Loc" is not updated (stays the same, regardless of the indexing)
                self.backward_clustering_structure[cluster_i] = cluster_i_structure

            return event_cluster_combination_balanced

        elif status == "continue":  # continue with the backward pass
            # update the backward_clustering_structure with the latest valid one
            # i.e. the new clustering structure

            self.backward_clustering_structure = self.clustering_structure
            event_cluster_combination_balanced = event_cluster_combination_below_loss  # same here
            return event_cluster_combination_balanced

        else:
            raise ValueError("Status code does not exist")

    def _check_event_model_constraints(self):

        """

        Checks the constraints the event model, i.e. event model 3, opposes on the input data.
        It uses the clustering_structure attribute, that is set in the _update_clustering() function.


        Returns
        -------
        checked_clusters : list
            List of triples (c1, c2, event_interval_t)
            with c1 being the identifier of the first cluster, c2 the second cluster
            in the c1 - c2 cluster-combination, that have passed the model
            checks. The event_interval_t are the indices of the datapoints in between the two
            clusters.

        """

        if self.debugging_mode is True:
            print("")
            print("Perform check 1 to find non noise cluster")

        # (1) it contains at least two clusters C1 and C2, besides the outlier cluster, and the outlier Cluster C0
        # can be non empty. (The noisy samples are given the the cluster -1 in this implementation of DBSCAN)
        n_with_noise_clusters = len(self.clustering_structure)
        n_non_noise_clusters = n_with_noise_clusters - 1 if -1 in self.clustering_structure.keys() else n_with_noise_clusters

        if self.debugging_mode is True:
            print("Number of non noise_clusters: " + str(n_non_noise_clusters))

        if n_non_noise_clusters < 2: #check (1) not passed
            return None

        # If check (1) is passed, continue with check (2)

        # (2) clusters C1 and C2 have a high temporal locality, i.e. Loc(Ci) >= 1 - temp_eps
        # i.e. there are at least two, non noise, clusters with a high temporal locality

        check_two_clustering = {} #store the clusters that pass the test in a new structure

        if self.debugging_mode is True:
            print("")
            print("Perform check 2 with temp locality greater than " + str(1 - self.temp_eps))
            print("Cluster | Temporal Locality")
            print("--------|----------------- ")
        for cluster_i, cluster_i_structure in self.clustering_structure.items():
            if cluster_i != -1: # for all non noise clusters

                if self.debugging_mode is True:
                   print(str(cluster_i) + "       | " + str(cluster_i_structure["Loc"]))

                if cluster_i_structure["Loc"] >= 1 - self.temp_eps: #the central condition of condition (2)
                    check_two_clustering[cluster_i] = cluster_i_structure


        if self.debugging_mode is True:
            print("Number of clusters that pass temporal locality epsilon(Check 2): " + str(n_non_noise_clusters) + " (min 2 clusters) ")

        if len(check_two_clustering) < 2:  #check (2) not passed
            return None

        # (3) two clusters C1 and C2 do not interleave in the time domain.
        # There is a point s in C1 for which all points n > s do not belong to C1 anymore.
        # There is also a point i in C2 for which all points n < i do not belong to C2 anymore.
        # i.e. the maximum index s of C1 has to be smaller then the minimum index of C2
        checked_clusters = []

        # We need to compare all pairs of clusters in order to find all pairs that fullfill condition (3)

        if self.debugging_mode is True:
            print("")
            print("Perform check 3 for all combinations ")

        cluster_combinations = itertools.combinations(check_two_clustering, 2) #get all combinations, without replacement

        for cluster_i, cluster_j in cluster_combinations: #for each combinations

            # the cluster with the smaller u, occurs first in time, we name it C1 according to the paper terminology here
            if check_two_clustering[cluster_i]["u"] < check_two_clustering[cluster_j]["u"]:
                #  #cluster_i is starting before cluster_j
                c1 = cluster_i
                c2 = cluster_j
            else: #then cluster_j is starting before cluster i
                c1 = cluster_j
                c2 = cluster_i

            # now we check if they are overlapping
            # the maximum index of C1 has to be smaller then the minimum index of C2, then no point of C2 is in C1
            # and the other way round, i.e all points in C1 have to be smaller then u of C2

            if check_two_clustering[c1]["v"] < check_two_clustering[c2]["u"]: #no overlap detected
                # if thee clusters pass this check, we can compute the possible event_interval_t (i.e. X_t)
                # for them. This interval possibly contains an event and is made up of all points between cluster c1
                # and cluster c2 that are noisy-datapoints, i.e. that are within the -1 cluster. THe noisy points
                # have to lie between the upper-bound of c1 and the lower-bound of c2
                if -1 not in self.clustering_structure.keys():
                    return None
                else:
                    c0_indices = self.clustering_structure[-1]["Member_Indices"] #indices in the -1 (i.e. noise) cluster


                #ASSUMPTION if there is no noise cluster, then the check is not passed.
                # No event segment is between the two steady state clusters then.
                # Any other proceeding would cause problems in all subsequent steps too.

                if self.debugging_mode is True:
                    print("No overlap between cluster " + str(c1) + " and " + str(c2) + " (i.e. a good Candidate)")
                    print("\tPotential event window (noise cluster indices:  " + str(c0_indices))
                    print("\tCluster 1 v: " + str(check_two_clustering[c1]["v"] ))
                    print("\tCluster 2 u: " + str(check_two_clustering[c2]["u"]))

                # check the condition, no overlap between the noise and steady state clusters allowed: u is lower, v upper bound
                condition = [(c0_indices > check_two_clustering[c1]["v"]) & (c0_indices < check_two_clustering[c2]["u"])]
                event_interval_t = c0_indices[condition]

                if self.debugging_mode is True:
                    print("\tEvent Intervall between cluster " + str(c1) + " and " + str(c2) + " with indices " + str(event_interval_t))

                # If the event_interval_t contains no points, we do not add it to the list too,
                # i.e. this combinations does not contain a distinct event interval between the two steady state
                # cluster sections.
                if len(event_interval_t) != 0:

                    checked_clusters.append((c1, c2, event_interval_t))

                else:
                    if self.debugging_mode is True:
                        print("Check 3: Event interval between the steady states was empty")
        # now all non interleaving clusters are in the check_three_clustering structure
        # If there are still at least two non-noise clusters in check_three_clustering, then check (3) is passed,
        # else not.


        if self.debugging_mode is True:
            print("Number of cluster-pairs that pass Check 3: " + str(len(checked_clusters)) + " (min 1 pair)")

        if len(checked_clusters) < 1:  # check (3) not passed


            return None


        return checked_clusters

    def compute_input_signal(self, voltage, current, period_length, original_non_log=False):
        """

        This functions uses the instantaneous voltage and current signals to compute the real (P) and reactive power (Q).
        The period_length has to divide length of the input signals evenly, i.e. no remainder.
        If this is not the case, an exception is raised.
        This is because the features are computed on a period-based approximation.

        The features are computed per 0.5 seconds, thus, two features per second are computed.

        ASSUMPTION: To reduce the variation in the input for the clustering, we logarthmize the data as the
        authors have done in other publications by them. In this paper, they do not detail the pre-processing and
        input releated hyperparameters like the window size.
        If you do not agree with this assumption, set the original_non_log parameter to True, and the signal
        will not be converted to log-scale.

        Args:
            voltage (ndarray): one-dimensional voltage array
            current (ndarray: one-dimensional current array
            period_length (int): length of a period for the given dataset.
                           example: sampling_rate=10kHz, 50Hz basefrequency,
                           10kHz / 50 = 200 samples / period
            original_non_log (bool): default: False, if set to true, the non-logarithmized data is returned.


        Returns:
            X (ndarray): Feature vector with active and reactive power, with shape(window_size_n,2).
                         The first component at time t is the active, the second one the reactive power.
                         The active and reactive power are per values per second.


        Parameters
        ----------
        voltage : ndarray
            Voltage signal, flat array
        current : ndarray
            Current signal, flat array
        period_length : int
            Number of samples that features are computed over
            Usually, length of a period for the given dataset
            example: sampling_rate=10kHz, 50Hz basefrequency,
            10kHz / 50 = 200 samples / period
        original_non_log : boolean, optional (default=False)
            False, if set to True, the non-logarithmized data is returned.

        Returns
        -------
        X : ndarray
            Feature vector with active and reactive power, with shape(window_size_n,2)
             The first component at time t is the active, the second one the reactive power
             The active and reactive power are per values per second

        """

        voltage, current = utils.check_X_y(voltage, current, force_all_finite=True, ensure_2d=False,allow_nd=False, y_numeric=True,
                  estimator="EventDet_Barsim_Sequential")

        period_length = int(period_length)

        compute_features_window = period_length*self.network_frequency/2 #compute features averaged over this timeframe

        # also ensure that the input length corresponds to full seconds
        if len(voltage) % (int(period_length) * self.network_frequency) != 0:
            raise ValueError("Ensure that the input signal can be equally divided with the period_length and full seconds! "
                             "The input has to correspond to full seconds, e.g. lengths such as 1.3 seconds are not permitted")
        self.original_non_log = original_non_log

        #Compute the active and reactive power using the Metrics class provided with this estimator
        Metrics = Electrical_Metrics()

        active_power_P= Metrics.active_power(instant_voltage=voltage, instant_current=current,
                                            period_length=compute_features_window) #values per second

        apparent_power_S = Metrics.apparent_power(instant_voltage=voltage, instant_current=current,
                                            period_length=compute_features_window) #values per second


        reactive_power_Q = Metrics.reactive_power(apparent_power=apparent_power_S, active_power=active_power_P)

        self.period_length = compute_features_window #used to convert the offsets back to the original data

        #Now we combine the two features into a signel vector of shape (window_size_n,2)
        X = np.stack([active_power_P, reactive_power_Q], axis=1)

        if original_non_log == False:

            X = np.log(X) #logarithmize the signal as described in the assumption

        return X

    def _convert_relative_offset(self, relative_offset, raw_period_length=None):
        """
        Convert the relative offset that is computed relative to the input of the algorithm, i.e. the return
        value of the compute_input_signal() function.

        This utility function can be used to adapt the offset back to the raw data.

        "To convert the relative offset, the method needs the period_length of the raw data. "
                             "Either call the compute_input_signal() method first (the parameter is set in there)"
                             "or provided it to the method "
        Parameters
        ----------
        relative_offset : int
            The offset that needs to be converted
        raw_period_length : int, optional (default=None)
            Length in samples of one period in the original (the target) raw data

        Returns
        -------
        target_offset : int
            Offset relative to the raw (target) input

        """

        if raw_period_length is None:
            raw_period_length = self.period_length

        if raw_period_length is None:
            raise ValueError("To convert the relative offset, the method needs the period_length of the raw data. "
                             "Either call the compute_input_signal() method first (the parameter is set in there)"
                             "or provided it to the method ")

        target_offset = relative_offset * raw_period_length

        return target_offset

    def _convert_index_to_timestamp(self, index, start_timestamp_of_window):
        """
        Function to convert an index that is relative to the start_timestamp of a window that was computed
        by the compute_input_signal function to a timestamp object.

        Args:
            index (int): index to convert, relative to the input window. Features that have been used to do the event
            detection and to get the index, have to be computed according to the compute_input_signal function.
            Otherwise the timestamps returned by this function can be wrong.

            start_timestamp_of_window(datetime): start timestamp of the window the event index is located in.
            network_frequency(int): basic network frequency
        Returns:

            event_timestamp (datetime)

        Parameters
        ----------
        index : int
            Index to convert, relative to the input window. Features that have been used to do the event
            detection and to get the index, have to be computed according to the compute_input_signal function
            Otherwise the timestamps returned by this function can be wrong
        start_timestamp_of_window : datetime.datetime
            Start timestamp of the window the event index is located in
            network_frequency(int): basic network frequency


        Returns
        -------
        event_timestamp : datetime.datetime
            Timestamp the index was converted to
        """

        seconds_since_start = index / 2# 2 values per second
        event_timestamp = start_timestamp_of_window + timedelta(seconds=seconds_since_start)

        return event_timestamp

    @staticmethod
    def score(ground_truth_events_p, detected_events_p, number_of_samples_in_dataset_p=None,
              tolerance_limit_sec_p=1, return_event_lists_p=False):
        """
        This function can be used to determined the values of the confusion matrix.

        It works as follows:
        For each of the detected events, we build an interval around the detected timestamp using the tolerance_limit_sec
            Then we check for each of the detected intervals if we find a ground truth event that falls within this intervall
             If this is the case we have found a True Positive Event (TP) :
               1. TP += 1
               2. Eliminate this event from the ground truth list, in order to: avoid double detection
               In the case there is more then one ground truth events that fits into the interval, we take the first one
             If this is not the case we have found a False Positive Event (FP):
               1. FP += 1

             After this procedure: all events that are left in the ground truth list have not been detected by the event detection algorithm
             (because they were not removed from the in the previous step)
             Hence those events are actually true, but are delcared to be false by the algorithm: They are False Negatives (FN)
             For each of those events left in the ground truth list: FN += 1

             Finally we determine the number of True Negative Events(TN):
             Lastly, when all the detected and groundtruth events have been processed, the TN are calculated by
             subtracting the TP, FN, and FP from the number of samples in the dataset, i.e., all the positions where an event could have happened
             The start and the end time of the dataset and the information about sampling rate is used to compute the total
             number of possible event timestamps.

        Parameters
        ----------
        ground_truth_events_p : list
            List of datetime objects, containing the ground truth event timestamps.

        detected_events_p : list
            List of datetime objects, containing the event detected by the event detector

        number_of_samples_in_dataset_p : int
            Number of samples, i.e. number of possible event positions in the dataset.
            When computing period wise features, like RMS, values use the corresponding number of samples.
            The value is used to compute the number of True Negative Events.

        tolerance_limit_sec_p : int, optional (default = 1)
            The tolerance limit that is used to determine the scores of the confusion matrix.

        return_events_list_p : boolean, optional (default = False)
            If set to True, the events that are TP, FP, and FN are returned in the results dictionary
        Returns
        -------

        if grid_search_mode is set to True, only the f1score is returned, otherwise:

        results: dictionary
            A dictionary containing the following keys:  tp, fp, fn, tn, f1score, recall, precision, fpp, fpr
            With being:
            tp = True Positives
            fp = False Positives
            fn = False Negatives
            tn = True Negatives
            f1score = F1-Score Metric
            recall = Recall Metric
            precision = Precision Metric
            fpp = False Positive Percentage Metric
            fpr = False Positive Rate Metric

            If return_events_list_p is set to True, the results also contain the following keys:
            tn_events = events that are tn
            fp_events = events that are fp
            fn_events = events that are fn

        """

        ground_truth_events = ground_truth_events_p #ground_truth_events_p.copy() #copy the events
        detected_events = detected_events_p.copy() #copy the events

        results = {}
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        TP_events = []
        FP_events = []
        FN_events = []


        # Compute the Tolerance Intervals around the detected events and iterate through the detected events
        for detected_event in detected_events:
            upper_limit = detected_event + timedelta(seconds=tolerance_limit_sec_p)
            lower_limit = detected_event - timedelta(seconds=tolerance_limit_sec_p)

            # Check which ground_truth events lie within this interval
            matching_ground_truth_events = [(index, ground_truth_event) for index, ground_truth_event in
                                            enumerate(ground_truth_events) if
                                            lower_limit <= ground_truth_event <= upper_limit]

            if len(matching_ground_truth_events) > 0:  # at least (minimum) one ground truth event was detected
                TP += 1
                first_index = matching_ground_truth_events[0][0]

                TP_events.append(ground_truth_events[first_index])
                # It can be possible that there are multiple ground truth events within this interval, hence,
                # we only consider the first one and remove it from the list
                del ground_truth_events[first_index]

            else:  # No Matching Event was found
                FP_events.append(detected_event)
                FP += 1

        # Use the remaining ground truth events to determine the number of FN
        FN = len(ground_truth_events)
        FN_events = ground_truth_events.copy()

        # Use the number of samples in the dataset to determine the TNs in the dataset
        # For BLUED we have data from 20/10/2011 11:58:32.623499 to 27/10/2011 14:10:40.0

        TN = number_of_samples_in_dataset_p - (TP + FP + FN)
        # Compute the final scores
        fpp = FP / (TP + FN)
        fpr = FP / (FP + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1score = (2 * precision * recall) / (precision + recall)

        results["tp"] = TP
        results["fp"] = FP
        results["fn"] = FN
        results["tn"] = TN
        results["f1score"] = f1score
        results["recall"] = recall
        results["precision"] = precision
        results["fpp"] = fpp
        results["fpr"] = fpr

        if return_event_lists_p is True:
            results["tp_events"] = TP_events
            results["fp_events"] = FP_events
            results["fn_events"] = FN_events

        return results

class EventDet_Liu_Ripple(BaseEstimator, ClassifierMixin):
    """
    Reference implementation for the following Event Detection algorithm:
                "A new Event Detection Technique for Residential Load Monitoring"

                by: Ming Liu, Jing Yong, Member, Xiaojing Wang, Jiaming Lu
                DOI: 10.1109/ICHQP.2018.8378820

    The algorithm implementation follows the general sklearn API.

    The authors describe two algorithms in their paper: an event detection and a subsequent event matching algorithm.
    Events that belong to the same device (turn on and switch-off events) are paired together in the second
    algorithm. This class only implements the first algorithm, the event detection algorithm.

    Input:
        Active Power (due to ASSUMPTION 1)

        ASSUMPTION 1: The authors do not clearly state the input signal to the algorithm.
        But they do mention they use power values. Furthermore they use the unit "W" and the variable "P" all over their paper.
        Hence, we assume the authors use active/real power as the input.
        Therefore, the "compute_input_signal" function returns the active power signal.

        ATTENTION: overlapping windows are required, because of the following reason.

        One shortcoming is, that the first and the last q values in this array can not be used in the ripple mitigation algorithm.
        Therefore, the data has to be streamed with overlapping windows to the algorithm.
        With an overlap of 2q. Why 2q?
        Because we can only compute the ripple mitigation for window_length - 2q samples.
        Therefore, we will do the following: (Except for the first window, where we have to leave out the first q samples.)
        When we overlap 2q samples, we can compute the CUMSUM windows for the q samples that
        we have left out before. We can only do this, if we have q samples the ones
        that have been left out. Hence, we need an overlap of 2q samples.

    Short Description of the algorithm and the input it requires:
        1. Apply a median filter algorithm to filter out noise in the signal
        2. Ripple mitigation via power difference.
            2.1 Compute the power delta at time i by: P_i = p_i+1 - p_i
            The aim of this step is to reduce the ripples in the power delta, to filter our flucutations
            that to not belong to events.
            2.2. Calculate a series of M around each datapoint j.
            M(j-m) = Sum of m delta values before P_j (including P_j)
            M(j+m) = Sum of m delta values after P_j (including P_j)

            We take the absolut value of each of the sums.

            m is varied here, from m=1,2,....,q.
            Hence, we get q M(j-m) sums before j, and q M(j+m) sums after j, overall 2q values.


            Then we take the minimum values ouf of all the 2q values, and set it to be
            the absolute value of P_j.
            By doing this for all datapoints we get a new, absolute, power delta signal.

        3. Apply a power threshold (authors use 10W) to the new absolute, delta power signal to
        identify events.



    """

    def __init__(self, median_filter_window=9, q_ripple_window_size=10, power_threshold=10, perform_input_order_checks=True, **kwargs):
        """
        Args:
            median_filter_window (int): size of the median filter window in samples
            default=9 samples with a sampling rate of 1Hz,
            q_ripple_window_size (int): size of the window for the ripple mitigation step (step 2)
            The window size is 2q+1, with q defaulting to 10.
            power_threshold (int): threshold applied to the absolute power delta signal to detect the events
            perform_input_order_check(bool): if True, then it is checked if the windows overlap with 2q (2*q_ripple_threshold
            values as demanded by the algorithm.
            **kwargs:

        Parameters
        ----------
        median_filter_window : int, optional (default=9)
            Size of the median filter window in samples
        q_ripple_window_size : int, optional (default=10)
            Size of the window for the ripple mitigation step (step 2)
            The window size is 2q+1, with q defaulting to 10.
        power_threshold : int, optional (default=10)
            Threshold applied to the absolute power delta signal to detect the events
        perform_input_order_checks: boolean, optional (default=True)
            If True, then it is checked if the windows overlap with 2q (2*q_ripple_threshold
        kwargs** : optional keyword arguments

        """

        self.median_filter_window = median_filter_window
        self.q_ripple_window_size = q_ripple_window_size
        self.power_threshold = power_threshold
        self.perform_input_order_checks = perform_input_order_checks
        self.order_safety_check = None

    def _input_order_check(self, X):
        """
        Checks if the input order matches the definition that is described in the input section of the documentation
        doc-string.


        Parameters
        ----------
        X : narray
            Input array, flat array, as computed by the compute_input_signal function

        Returns
        -------

        """

        if self.order_safety_check is not None: #in the first call of predict it is None
            # the last 2q values of the preceding window should be
            # equal to the first 2q values of the new window

            end = self.q_ripple_window_size * 2 # 2q

            X_test = X[:end] #the last q values should be the first q v

            if not np.array_equal(self.order_safety_check, X_test):
                #if the first 10 values are not the same
                raise ValueError("Please stick to the input order described in the documentation! The first 2q values "
                                 "of your input do not match the last 2q values from the previous window as required. "
                                 "If you do not want to see "
                                 "this message anymore, set perform_input_order_checks=False during initialization!")

    def fit(self):
        """
        Does nothing

        """
        self.is_fitted = True

    def predict(self, X):
        """
        Perform the event detection on the data provided

        Parameters
        ----------
        X : ndarray
            dim=(1,) input data, the active power values computed by the compute_input_signal function

        Returns
        -------
        event_indices : list
            List of the event indices

        """

        # Check if fit was called before
        check_is_fitted(self, ["is_fitted"])

        # Check the input
        # Perform general tests on the input array

        utils.assert_all_finite(X)
        X = utils.as_float_array(X)

        # Perform checks whether the input was fed in the correct way to the algorithm
        if self.perform_input_order_checks == True:
            self._input_order_check(X)

        # Run the median filter
        X_median_filtered = self._median_filter(X)

        # Run the ripple mitigation
        X_ripple_mitigated = self._ripple_mitigation(X_median_filtered)

        # Apply the power threshold to detect the events
        event_indices = np.where(X_ripple_mitigated >= self.power_threshold)[0]
        event_indices += self.ripple_offset

        # Set the data from this window for the next input order check
        offset = len(X) - (self.q_ripple_window_size * 2) #2q values
        self.order_safety_check = X[offset:]
        assert len(self.order_safety_check) == self.q_ripple_window_size * 2


        return event_indices

    def _median_filter(self, X):
        """
        Apply a median filter to remove noise from the signal.
        It replaces each record in the signal by its neighboring records.
        How many neighbors are used, is determined by the "window" size.

        If the window size is set as 2n + 1, power impulses lasting less than n records will be
        eliminated. So it should not be too long, or to short.

        The window size can be determined by looking at the starting times and the operations times of typical
        appliances in the environment .
        Based on their investigations a window size of 9 is preset.

        In the boundary cases, the first and the last records are repeated to fill the window.

        Parameters
        ----------
        X : ndarray
            dim=(1,) input data, the active power values computed by the compute_input_signal function

        Returns
        -------
        X_median_filtered : ndarray
            dim=(1,) median filtered input signal

        """


        # center=True, leads to the value being computed for
        # the central value of the window. Hence, with a window of 9, we get 4 to the left and 4 to the right always.
        X_median_filtered = pd.Series(X).rolling(self.median_filter_window, center=True).median().values


        # replace the non-values at the border of the array
        # the first and the last values are repeated to fill the values in the boundary cases
        non_nan_indices = np.argwhere(~np.isnan(X_median_filtered)) #get indices where no NaN values (~ inverts)
        left_boundary = np.min(non_nan_indices) #get the index of the first non nan value
        right_boundary = np.max(non_nan_indices) #get the index of the last non nan value

        X_median_filtered[:left_boundary] = X_median_filtered[left_boundary] #set the first nan values to the first value
        X_median_filtered[right_boundary:] = X_median_filtered[right_boundary] #set the last nan values to the last value

        assert len(X_median_filtered) == len(X)

        return X_median_filtered

    def _ripple_mitigation(self, X_median_filtered):
        """
        Ripple mitigation algorithm.

        1. Compute difference (power_delta), between consecutive power samples
        2. For each sample i:
            2.1 Take q windows to the left and q windows to the right of sample  i
            2.2 Sum the power_deltas for each of the 2q windows
            This is equal to building the CUMSUM of q values in each direction around i.
            2.3 Take the absolute values of the sums for each of the 2q sums
            This is equal to taking the absolute value of the CUMSUM of the q values around i.
            2.4 Take the minimum value among all the 2q sums and set the value to the new value of i
            This is equal to taking the minimum among the CUMSUM values to the right and to the left

        By doing this, we obtain an absolute power_delta signal.

        One shortcoming is, that the first and the last q values in this array can not be used in the ripple mitigation algorithm.
        Therefore, the data has to be streamed with overlapping windows to the algorithm.
        With an overlap of 2q. Why 2q?
        Because we can only compute the ripple mitigation for window_length - 2q samples.
        Therefore, we will do the following: (Except for the first window, where we have to leave out the first q samples.)
        When we overlap 2q samples, we can compute the CUMSUM windows for the q samples that
        we have left out before. We can only do this, if we have q samples the ones
        that have been left out. Hence, we need an overlap of 2q samples.


        Parameters
        ----------
        X_median_filtered : ndarray
            dim=(1,) median filtered input signal, as returned from the _median_filter function

        Returns
        -------
        absolute_power_delta : ndarray
            dim=(1,) ripple mitigated, absolute power delta values

        """



        power_delta = np.diff(X_median_filtered)
        power_delta = np.concatenate([power_delta, [power_delta[-1]]]) #duplicate the last value, to have the same array length again

        # compute indices to start and end the CUMSUM computations.
        # we need q samples to the left and q samples to the right of this window
        start_idx = self.q_ripple_window_size #is the index 10, then there are 10 values to the left.
        end_idx = len(power_delta) - self.q_ripple_window_size

        #the indices match the starting of the absolute_power_delta
        #that will be created in the following.

        absolute_power_delta = []
        for i_idx, i_value in enumerate(power_delta[start_idx:end_idx]):
            i_idx = i_idx + start_idx #add start_idx


            cumsum_area_left = power_delta[i_idx-self.q_ripple_window_size: i_idx+1] #include the i value in the sum.
            cumsum_area_left = cumsum_area_left[::-1]
            # it is reversed with -1, because the sums in the paper are made from right to left for
            # the q points left of i
            cumsums_left = np.cumsum(cumsum_area_left)

            cumsum_area_right = power_delta[i_idx: i_idx + self.q_ripple_window_size+1]
            cumsums_right = np.cumsum(cumsum_area_right)

            # The paper demands that the first sum is value i plus the next value.
            # The cumsum also did just take value i as the sum of itself as the starting value, hence we need to throw it way
            cumsums_left = cumsums_left[1:] # the first value is j here too, because we have reversed it above
            cumsums_right = cumsums_right[1:]

            # No need to reverse the cumsums left again, order does not matter from now for the sums.
            cumsums = np.concatenate([cumsums_left, cumsums_right], axis=0)
            cumsums = np.abs(cumsums) #absolute values of the sums

            absolute_power_delta.append(np.min(cumsums))

        # set the offset used here, in order to be able to adjust the detected event indices

        # to the original indices
        self.ripple_offset = start_idx
        return np.array(absolute_power_delta)

    def _convert_relative_offset(self, relative_offset, raw_period_length=None):
        """
        Convert the relative offset that is computed relative to the input of the algorithm, i.e. the return
        value of the compute_input_signal() function.

        This utility function can be used to adapt the offset back to the raw data.

        "To convert the relative offset, the method needs the period_length of the raw data. "
                             "Either call the compute_input_signal() method first (the parameter is set in there)"
                             "or provided it to the method "
        Parameters
        ----------
        relative_offset : int
            The offset that needs to be converted
        raw_period_length : int, optional (default=None)
            Length in samples of one period in the original (the target) raw data

        Returns
        -------
        target_offset : int
            Offset relative to the raw (target) input

        """

        if raw_period_length is None:
            raw_period_length = self.period_length

        if raw_period_length is None:
            raise ValueError("To convert the relative offset, the method needs the period_length of the raw data. "
                             "Either call the compute_input_signal() method first (the parameter is set in there)"
                             "or provided it to the method ")

        target_offset = relative_offset * raw_period_length

        return target_offset

    def _convert_index_to_timestamp(self, index, start_timestamp_of_window, network_frequency):
        """
       Function to convert an index that is relative to the start_timestamp of a window that was computed
       by the compute_input_signal function to a timestamp object.

       Args:
           index (int): index to convert, relative to the input window. Features that have been used to do the event
           detection and to get the index, have to be computed according to the compute_input_signal function.
           Otherwise the timestamps returned by this function can be wrong.

           start_timestamp_of_window(datetime): start timestamp of the window the event index is located in.
           network_frequency(int): basic network frequency
       Returns:

           event_timestamp (datetime)

       Parameters
       ----------
       index : int
           Index to convert, relative to the input window. Features that have been used to do the event
           detection and to get the index, have to be computed according to the compute_input_signal function
           Otherwise the timestamps returned by this function can be wrong
       start_timestamp_of_window : datetime.datetime
           Start timestamp of the window the event index is located in
           network_frequency(int): basic network frequency


       Returns
       -------
       event_timestamp : datetime.datetime
           Timestamp the index was converted to
       """

        seconds_since_start = index * (1 / network_frequency)
        event_timestamp = start_timestamp_of_window + timedelta(seconds=seconds_since_start)

        return event_timestamp

    def compute_input_signal(self, voltage, current, period_length):
        """
        This functions uses the instantaneous voltage and current signals to compute the real/active (P).
        The period_length has to divide length of the input signals evenly, i.e. no remainder.
        If this is not the case, an exception is raised.
        This is because the features are computed on a period-based approximation




        Parameters
        ----------
        voltage : ndarray
            Voltage array, flat array
        current : ndarray
            Current array, flat array
        period_length : int
            Number of samples that features are computed over
            Usually, length of a period for the given dataset
            example: sampling_rate=10kHz, 50Hz basefrequency,
            10kHz / 50 = 200 samples / period


        Returns
        -------
        active_power_P : ndarray
            Feature vector with real/active with shape(window_size_n).

        """


        voltage, current = utils.check_X_y(voltage, current, force_all_finite=True, ensure_2d=False, allow_nd=False,
                                           y_numeric=True,
                                           estimator="EventDet_Barsim_Sequential")

        period_length = int(period_length)

        if len(voltage) % int(period_length) != 0:
            raise ValueError("Ensure that the input signal can be equally divided with the period_length!")


        # Compute the active and reactive power using the Metrics class provided with this estimator
        Metrics = Electrical_Metrics()

        active_power_P = Metrics.active_power(instant_voltage=voltage, instant_current=current,
                                              period_length=period_length)

        self.period_length = period_length  # used to convert the offsets back to the original data

        # Now we combine the two features into a signel vector of shape (window_size_n,2

        return active_power_P

    @staticmethod
    def event_list_postprocessing(events_detected, postprocessing_limit):
        """
        Postprocess the detected events list.
        All events that are detected within one second of an detected event are discarded.
        As the Liu event detector is very sensible, this improves the results remarkably
        and is a legit step for fine tuning the algorithm.


        Parameters
        ----------
        events_detected : list
            List of detected events, as datetime.datetime objects
        postprocessing_limit : int
            Value in seconds that determines the range in which events
            are discarded.

        Returns
        -------
        events_detected_postprocessed : ndarray
            List of the post-processed events

        """

        events_detected_postprocessed = [events_detected[0]]  # add the first element to begin with
        for event in events_detected:
            # for every event, check if it is within one second of the last element in the already
            # post processed ones. If not, add them to the postprocessed events list
            last_postprocessed_event = events_detected_postprocessed[-1]
            diff = abs(event - last_postprocessed_event)

            if diff.total_seconds() > postprocessing_limit:
                events_detected_postprocessed.append(event)

        events_detected_postprocessed = np.array(events_detected_postprocessed)

        return events_detected_postprocessed
    @staticmethod
    def score(ground_truth_events_p, detected_events_p, number_of_samples_in_dataset_p=None,
              tolerance_limit_sec_p=1, return_event_lists_p=False):
        """
        This function can be used to determined the values of the confusion matrix.

        It works as follows:
        For each of the detected events, we build an interval around the detected timestamp using the tolerance_limit_sec
            Then we check for each of the detected intervals if we find a ground truth event that falls within this intervall
             If this is the case we have found a True Positive Event (TP) :
               1. TP += 1
               2. Eliminate this event from the ground truth list, in order to: avoid double detection
               In the case there is more then one ground truth events that fits into the interval, we take the first one
             If this is not the case we have found a False Positive Event (FP):
               1. FP += 1

             After this procedure: all events that are left in the ground truth list have not been detected by the event detection algorithm
             (because they were not removed from the in the previous step)
             Hence those events are actually true, but are delcared to be false by the algorithm: They are False Negatives (FN)
             For each of those events left in the ground truth list: FN += 1

             Finally we determine the number of True Negative Events(TN):
             Lastly, when all the detected and groundtruth events have been processed, the TN are calculated by
             subtracting the TP, FN, and FP from the number of samples in the dataset, i.e., all the positions where an event could have happened
             The start and the end time of the dataset and the information about sampling rate is used to compute the total
             number of possible event timestamps.

        Parameters
        ----------
        ground_truth_events_p : list
            List of datetime objects, containing the ground truth event timestamps.

        detected_events_p : list
            List of datetime objects, containing the event detected by the event detector

        number_of_samples_in_dataset_p : int
            Number of samples, i.e. number of possible event positions in the dataset.
            When computing period wise features, like RMS, values use the corresponding number of samples.
            The value is used to compute the number of True Negative Events.

        tolerance_limit_sec_p : int, optional (default = 1)
            The tolerance limit that is used to determine the scores of the confusion matrix.

        return_events_list_p : boolean, optional (default = False)
            If set to True, the events that are TP, FP, and FN are returned in the results dictionary
        Returns
        -------

        if grid_search_mode is set to True, only the f1score is returned, otherwise:

        results: dictionary
            A dictionary containing the following keys:  tp, fp, fn, tn, f1score, recall, precision, fpp, fpr
            With being:
            tp = True Positives
            fp = False Positives
            fn = False Negatives
            tn = True Negatives
            f1score = F1-Score Metric
            recall = Recall Metric
            precision = Precision Metric
            fpp = False Positive Percentage Metric
            fpr = False Positive Rate Metric

            If return_events_list_p is set to True, the results also contain the following keys:
            tn_events = events that are tn
            fp_events = events that are fp
            fn_events = events that are fn

        """

        ground_truth_events = ground_truth_events_p #ground_truth_events_p.copy() #copy the events
        detected_events = detected_events_p.copy() #copy the events

        results = {}
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        TP_events = []
        FP_events = []
        FN_events = []


        # Compute the Tolerance Intervals around the detected events and iterate through the detected events
        for detected_event in detected_events:
            upper_limit = detected_event + timedelta(seconds=tolerance_limit_sec_p)
            lower_limit = detected_event - timedelta(seconds=tolerance_limit_sec_p)

            # Check which ground_truth events lie within this interval
            matching_ground_truth_events = [(index, ground_truth_event) for index, ground_truth_event in
                                            enumerate(ground_truth_events) if
                                            lower_limit <= ground_truth_event <= upper_limit]

            if len(matching_ground_truth_events) > 0:  # at least (minimum) one ground truth event was detected
                TP += 1
                first_index = matching_ground_truth_events[0][0]

                TP_events.append(ground_truth_events[first_index])
                # It can be possible that there are multiple ground truth events within this interval, hence,
                # we only consider the first one and remove it from the list
                del ground_truth_events[first_index]

            else:  # No Matching Event was found
                FP_events.append(detected_event)
                FP += 1

        # Use the remaining ground truth events to determine the number of FN
        FN = len(ground_truth_events)
        FN_events = ground_truth_events.copy()

        # Use the number of samples in the dataset to determine the TNs in the dataset
        # For BLUED we have data from 20/10/2011 11:58:32.623499 to 27/10/2011 14:10:40.0

        TN = number_of_samples_in_dataset_p - (TP + FP + FN)
        # Compute the final scores
        fpp = FP / (TP + FN)
        fpr = FP / (FP + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1score = (2 * precision * recall) / (precision + recall)

        results["tp"] = TP
        results["fp"] = FP
        results["fn"] = FN
        results["tn"] = TN
        results["f1score"] = f1score
        results["recall"] = recall
        results["precision"] = precision
        results["fpp"] = fpp
        results["fpr"] = fpr

        if return_event_lists_p is True:
            results["tp_events"] = TP_events
            results["fp_events"] = FP_events
            results["fn_events"] = FN_events

        return results

class Electrical_Metrics:
    """
    Class that contains several functions to compute (approximately) diverse Electrical metrics:

    active_power
    apparent_power
    reative_power
    power_factor
    voltage_current_rms
    single_rms

    """
    def __init__(self):
        pass

    def active_power(self,instant_voltage, instant_current,period_length):
        """
        Active or Real power is the average of instantaneous power.
        P = Sum ( i[n] * v[n] ) / N )
        First we calculate the instantaneous power by multiplying the instantaneous
        voltage measurement by the instantaneous current measurement. We sum the
        instantaneous power measurement over a given number of samples and divide by
        that number of samples.


        Parameters
        ----------
        instant_voltage : ndarray
            Instantaneous Voltage, flat array
        instant_current : ndarray
            Instantaneous Current, flat array
        period_length : int
            Number of samples the features are computed over

        Returns
        -------
        active_power : ndarray
            Active Power array

        """

        instant_current = np.array(instant_current).flatten()
        instant_voltage = np.array(instant_voltage).flatten()

        if len(instant_current) == len(instant_voltage):
            instant_power = instant_voltage * instant_current
            period_length = int(period_length)

            active_power = []
            for i in range(0, len(instant_power), period_length):
                if i + period_length <= len(instant_power):
                    signal_one_period = instant_power[i:int(i + period_length)]
                    active_power_one_period = np.mean(signal_one_period )
                    active_power.append(active_power_one_period)
            active_power = np.array(active_power)
            return active_power

        else:
            raise ValueError("Signals need to have the same length")

    def apparent_power(self, instant_voltage,instant_current,period_length):
        """
        Compute apparent power:
        S = Vrms * Irms

        Parameters
        ----------
        instant_voltage : ndarray
            Instantaneous Voltage, flat array
        instant_current : ndarray
            Instantaneous Current, flat array
        period_length : int
            Number of samples the features are computed over

        Returns
        -------
        apparent_power : ndarray
            Apparent Power array

        """
        if len(instant_current) == len(instant_voltage):

            rms_voltage = self.compute_single_rms(instant_voltage,period_length)
            rms_current = self.compute_single_rms(instant_current,period_length)
            apparent_power = rms_voltage * rms_current
            return apparent_power

        else:
            raise ValueError("Signals need to have the same length")

    def reactive_power(self,apparent_power,active_power):
        """
        Compute reactive power:
        Q = sqrt(S^2 - P^2)

        Parameters
        ----------
        apparent_power : ndarray
            Apparent power, flat array
        active_power : ndarray
            Active power, flat array

        Returns
        -------
        reactive_power : ndarray
            Reactive power, flat array

        """

        if len(apparent_power) == len(active_power):
            reactive_power = np.sqrt((apparent_power * apparent_power) - (active_power * active_power))
            return reactive_power
        else:
            raise ValueError("Signals need to have the same length")


    def compute_power_factor(self,apparent_power,active_power):
        """
        Compute power factor:
        PF = P / S

        Parameters
        ----------
        apparent_power : ndarray
            Apparent power, flat array
        active_power : ndarray
            Active power, flat array

        Returns
        -------
        power_factor : float
            Power factor

        """

        power_factor = active_power / apparent_power
        return power_factor


    def compute_voltage_current_rms(self, voltage, current, period_length):
        """
        Compute Root-Mean-Square (RMS) values for the provided voltage and current.

        Parameters
        ----------
        voltage : ndarray
            Instantaneous Voltage, flat array
        current : ndarray
            Instantaneous Current, flat array
        period_length : int
            Number of samples the features are computed over

        Returns
        -------
        voltage_rms : ndarray
            Voltage RMS values
        current_rms : ndarray
            Current RMS values

        """
        period_length = int(period_length)
        voltage_rms = self.compute_single_rms(voltage, period_length)
        current_rms = self.compute_single_rms(current, period_length)
        return voltage_rms, current_rms


    def compute_single_rms(self,signal,period_length):
        """
        Compute Root-Mean-Square (RMS) values for the provided signal.

        Parameters
        ----------
        signal : ndarray
            Instantaneous Voltage OR Current flat array

        period_length : int
            Number of samples the features are computed over

        Returns
        -------
        signal_rms : ndarray
            RMS values of signal

        """
        rms_values = []
        period_length = int(period_length)
        for i in range(0, len(signal), period_length):
            if i + period_length <= len(signal):
                signal_one_period = signal[i:int(i + period_length)]
                rms_one_period = np.sqrt(np.mean(np.square(signal_one_period))) #rms
                rms_values.append(rms_one_period)
        return np.array(rms_values)