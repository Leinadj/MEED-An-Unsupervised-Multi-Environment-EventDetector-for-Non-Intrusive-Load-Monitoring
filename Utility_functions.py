import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre
import sklearn.model_selection as skmose
import random
import os
import platform
from pathlib import Path
import matplotlib
if platform.system() == "Linux": #for matplotlib on Linux
    matplotlib.use('Agg')

import abc

import pdb

import glob

from datetime import timedelta

from io import StringIO
from datetime import datetime

from copy import deepcopy

#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

class BaseDataset(object,metaclass=abc.ABCMeta):

    #TODO in die init methode der abstrakten klasse noch ein paar Pflichtfelder mehr mit reinmachen!!
    @abc.abstractmethod
    def populate(self):
        """
        Populates the Dataset Object: i.e. Loades its full Metadata and all the filenames into memory for further use
        ATTENTION: Can take a bit and can be memory consuming, if working with single samples this is not necessary
        Returns:

        """


        raise NotImplementedError("define a populate method")
    @abc.abstractmethod
    def load_File(self,file_path,preprocessing_func):
        """

          Args:
              file_path: path to the file that is loadad
              class_label: the label corresponding to the file
              preprocessing_func: a function used to preprocess the data (two arguments one for voltage and one for current)

          Returns:

          """
        raise NotImplementedError("A Dataset has to implement a load_File method")

    @abc.abstractmethod
    def load_Metadata(self,file_path):
        #The identifier is the filename!!! it has to be unique!!
        raise NotImplementedError("A Dataset has to implement a load_Metadata method")

    @abc.abstractmethod
    def load_Batch(self):
        raise NotImplementedError("A Dataset has to implement a load_full_Dataset method")

    @abc.abstractmethod
    def save_File(self,filepath,voltage,current):
        raise NotImplementedError("A Dataset has to implement a save_File method")


    @abc.abstractmethod
    def calibrate_Sample(self,voltage,current):
        raise NotImplementedError("Every dataset needs a calibration methode")

    def split_Data(self,data_split_proportions_p,cross_val_p = False):
        #set the parameter self.training_files, self.test_files, self.validation_files wenn cross_val_p == False
        #wenn cross_val == True: safe cross_val_object
        #evtl. schon beim init die parameter, sowie den filename parameter der beim populate gefüllt wird auf None initialisieren
        #every filename with a central flag = True hat einen liste mit den filenames / appliance ids die davon abgeleited sind in den metadaten
        #Windows mit einem central flag sind die windows auf denen die augmetned daten basieren! (i.e. they are the original windows)
        #Unabhängig vom Datensatz
        #der Filename / die appliance id /... der key des metadata dict ist immer der unique identifier über den sich die Verbindung herstellen l#sst


        # if it is not a central window, append it to the corresponding central windows reference array

        #the data_split_proportions_p ist ein dictionary (siehe Experiment Klasse in der __init__ methode wie das auszusehen hat


        training = []
        testing = []
        validation = []


        if self.metadata == None:
            raise ValueError("The metadata is not populated yet, please use the populate method on the dataset object first")


        #Perform augmented split if the dataset is already augmented on disk
        if self.augmented == True:

            #1. fill the reference list for every central window
                #all augemnted files begin with the identifier of the central window --> not dataset dependent, delimeter is always a "_"

            central_sample_list = [] #needed for the split in step 2

            for central_identifier in self.metadata.keys():
                if central_identifier["central"]:

                    central_sample_list.append(central_identifier)

                    # loop over all others
                    for sample_identifier in self.metadata.keys():
                        if sample_identifier["central"] == False and sample_identifier.startswith(str("central_identifier"+"_")) == True:
                            #if the sample is an derived / augmented one and if it belongs to the central_identifier
                            self.metadata[central_identifier]["derived_appliances"].append(sample_identifier)


            #2. shuffle and split the central windows
            central_sample_list_training, central_sample_list_rest = skmose.train_test_split(central_sample_list,train_size=data_split_proportions_p["training"],shuffle=True,random_state=10)
            central_sample_list_validation, central_sample_list_test = skmose.train_test_split(central_sample_list_rest,train_size=data_split_proportions_p["validation"],shuffle=True,random_state=10)


            #3. append the corresponding non central_windows to the list and shuffle it again
            for central_training_sample in central_sample_list_training:
                training.append(central_training_sample)
                training.extend(self.metadata[central_training_sample]["derived_appliances"])

            for central_validation_sample in central_sample_list_validation:
                validation.append(central_validation_sample)
                validation.extend(self.metadata[central_validation_sample]["derived_appliances"])

            for central_test_sample in central_sample_list_test:
                testing.append(central_test_sample)
                testing.extend(self.metadata[central_test_sample]["derived_appliances"])

        if self.augmented == False:
            training, rest = skmose.train_test_split(self.filenames, traing_size=data_split_proportions_p["training"],shuffle=True,random_state=10)
            validation, testing = skmose.train_test_split(rest,test_size=data_split_proportions_p["validation"],shuffle=True,random_state=10)

        #TODO placeholder für cross_validation split

    def create_Batches(self,batch_size, filenames, batch_number = None):
        """
        Creates Batch from a list of filenames, either by handing over a fixed batch_size:
        Each of the batches then has a batch size of batch_size, or by handy over a batch_number
        Then a fixed number of batches (batch_number) is created. The size of each individual batch then depends on the size of the dataset
        The batches are computed uisng floor devision: ATTENTION: in this case some samples might be lost!
        Args:
            batch_size:
            filenames:
            batch_number:
            shuffle:

        Returns:
            ndarray of batch ndarrays:
        """
        #Data is already shuffled
        #TODO evtl. noch nen shuffle implementieren
        if batch_number == None:
            number_of_files = len(filenames)
            batch_number = number_of_files // batch_size #floor division

            #TODO testen ob der Split einigermaßen gut hinhaut

        batches = np.array_split(np.array(filenames).flatten(),int(batch_number))


        return batches

class BLUED_Dataset(BaseDataset):

    def __init__(self, name, dataset_location, event_file_path, samplerate, training_folders=1):
        self.dataset_name = name
        self.dataset_location = dataset_location
        self.samplerate = samplerate
        self.days = np.arange(20, 28)
        self.event_file_path = event_file_path
        self.start_recording_time = datetime.strptime("2011/10/20-11:58:32.62349", '%Y/%m/%d-%H:%M:%S.%f')

    def populate(self):
        self.dataset_location

    def load_File(self, file_path, preprocessing_func=None, type="original_csv",_run=None,convert_to_timestamps=False,phase="all"):
        """
        The time steps are measured in seconds, a new sample is measured every measurement_steps steps

        Args:
            file_path:
            preprocessing_func:
            type:
            _run:
            convert_to_timestamps:
            phase:

        Returns:
            data_df: original columns if phase=="all" else colums are just "Current" and "Voltage" --> already for the matching phase! (* - 1 done for B)
            file_info: dictionary

        """



        if type == "original_csv":


            with open(file_path, 'r') as f:

                data_txt = f.read()

                lines = data_txt.splitlines()



                #TODO hier passiert der Fehler, reference time ist nicht überall gleich
                data_txt = data_txt.split("***End_of_Header***")
                reference_time = data_txt[0].split("Date,")[1][:11].replace("\n","") +"-"+ data_txt[0].split("Time,")[1][:15]
                reference_time = datetime.strptime(reference_time, '%Y/%m/%d-%H:%M:%S.%f')


                data_time_str = data_txt[1].split("Time,")[1]
                data_time_str = data_time_str.split(',')

                data_day_str = data_txt[1].split("Date,")[1]
                data_day_str = data_day_str.split(',')

                day_str = data_day_str[0]  # just the first on is enoguh
                time_str = data_time_str[0][:15]  # same for time
                date = day_str + "-" + time_str
                start_date_time = datetime.strptime(date, '%Y/%m/%d-%H:%M:%S.%f')

                # number of samples * file_delta

                filename = Path(file_path).name  # get the file bath %TODO debug

                samples = data_txt[1].split("Samples,")[1].split(",")[0:3][0]
                samples = int(samples)

                values_str = data_txt[-1]
                values_str = values_str[values_str.index("X_Value"):]

                measurement_steps = data_txt[1].split("Delta_X")[1].split(",")[0:3]
                measurement_steps = [float(x) for x in measurement_steps if x != ""]
                measurement_steps = measurement_steps[0]

                # TODO DEBUG THE DELTA float thing


                #TODO TIMING MACHEN
                data_df = pd.read_csv(StringIO(values_str), usecols=["X_Value", "Current A", "Current B", "VoltageA"])


                data_df.dropna(inplace=True,how="any")
                #5051 und 5055 sind korrupt

                if "5051" in filename or "5055" in filename:

                    pass
                else:
                    pass
                file_duration = data_df.tail(1)["X_Value"].values[0]
                file_duration = float(file_duration)

                file_duration = timedelta(seconds=file_duration)
                end_date_time = reference_time + file_duration

                file_duration = end_date_time - start_date_time


                if convert_to_timestamps == True:
                    data_df["TimeStamp"] = data_df["X_Value"].apply(lambda x: timedelta(seconds=x) + reference_time)
                    data_df.drop(columns=["X_Value"],inplace=True)
                    data_df.set_index("TimeStamp",inplace=True)

                file_info = {"Filepath": file_path, "Filename": filename, "samples": samples,
                             "file_start": start_date_time, "file_duration": file_duration, "file_end": end_date_time,
                             "measurement_steps": measurement_steps,"reference_time":reference_time}

                if phase.lower() != "all":
                    if phase.lower() == "a":
                        data_df["Current"] = data_df["Current A"]
                        data_df["Voltage"] = data_df["VoltageA"]
                    elif phase.lower() == "b":
                        data_df["Current"] = data_df["Current B"]
                        data_df["Voltage"] = data_df["VoltageA"].values * -1
                    else:
                        raise ValueError("The phase provided does not exist")

                    data_df.drop(columns=['Current A', 'Current B',"VoltageA"],inplace=True)

                return data_df, file_info

    def _load_file_metadata(self, file_path):
        """
        Only used to load the metadata of a file
        Returns:



        """

        with open(file_path, 'r') as f:
            data_txt = f.read()
            lines = data_txt.splitlines()
            last_line = lines[-1]
            file_duration = float(last_line.split(",")[0])
            file_duration = timedelta(seconds=file_duration)

            data_txt = data_txt.split("***End_of_Header***")

            data_time_str = data_txt[1].split("Time,")[1]
            data_time_str = data_time_str.split(',')

            data_day_str = data_txt[1].split("Date,")[1]
            data_day_str = data_day_str.split(',')

            """
            file_duration = data_txt[1].split("X0")[1].split(",")[0:3]
            file_duration = [float(x) for x in file_duration]
            """

            # number of samples * file_delta

            # signals = ["Current A", "Current B", "Voltage A"]
            file_info = {}

            day_str = data_day_str[0]  # just the first on is enoguh
            time_str = data_time_str[0][:15]  # same for time

            date = day_str + "-" + time_str
            start_date_time = datetime.strptime(date, '%Y/%m/%d-%H:%M:%S.%f')
            end_date_time = start_date_time + file_duration

            file_info["file_start"] = start_date_time
            file_info["file_end"] = end_date_time
            file_info["file_duration"]

            filename = Path(file_path).name  # get the file bath %TODO debug
            file_info["Filepath":file_path]
            file_info["Filename":filename]

            return file_info

    def _load_all_file_times(self):
        subfolders_list = next(os.walk(self.dataset_location))[1]

        subfolder_list = [folder for folder in subfolders_list if folder.startswith("location_001_dataset_")]

        col_names = ['Filename', 'Filepath', 'file''Current A Start', 'Current A End', 'Current A Duration',
                     'Current B Start', 'Current B End', 'Current B Duration', 'Voltage A Start', 'Voltage A End',
                     'Voltage B Duration']
        time_df = pd.DataFrame(columns=col_names)

        for folder_index, folder in enumerate(subfolder_list):
            files = glob.glob(os.path.join(self.dataset_location, folder, "location_001_ivdata_*.txt"))
            folder_time_df = pd.DataFrame(columns=col_names)

            for file_index, file_path in enumerate(files):
                file_data = self._load_file_metadata(file_path)

                folder_time_df.append(file_data)
            time_df.append(folder_time_df)
            del folder_time_df

        return time_df

    def _determine_file(self, timestamp):

        pass

    def _load_events(selfs, file_path, phase):

        events_df = pd.read_csv(file_path, header=0)
        events_df["Timestamp"] = pd.to_datetime(events_df["Timestamp"],
                                                format='%m/%d/%Y %H:%M:%S.%f')  # 10/20/2011 15:45:54.590

        events_df["StartFileIndex"] = np.nan
        events_df["StartFileFilename"] = np.nan

        events_df["EndFileIndex"] = np.nan
        events_df["EndFileFilename"] = np.nan
        return events_df

    def _load_labels_mapping(selfs, file_path, phase):

        if phase != "A" or phase != "B":
            raise ValueError("Please Provide a valid phase!")
        labels_dict = {}
        with open(file_path, 'r') as f:
            labels_list = f.read().splitlines()
            for line in labels_list:
                tokens = line.split(" - ")
                if tokens[0].endswith(phase):
                    labels_dict[tokens[0].replace(phase, "")] = tokens[1]

    def produce_event_files(self, duration, phase="B", voltage=False):
        """
        Only loads current signals, compares the timestamps to the start of the corresponding phase
        Args:
            duration:
            phase:

        Returns:

        """
        # TODO save the event files --> cut druation before and duration afer the event after the window: 120000*5=
        file_time_df = self._load_all_file_times()
        events_df = self._load_events(self.event_file_path, "B")
        file_time_df.set_index([str("Current " + phase + " Start")], inplace=True)
        file_time_df.sort_index()  # sort ascending

        # TODO auch voltage mitverwenden

        # NOTE: Determine the file where the event starts - duration

        # TODO center around event timestamp

        # Make the column for the corresponding phase the new index
        for row_index, row in events_df.iterrows():  # for each event
            event_timestamp = row["Timestamp"]

            pre_event_timestamp = event_timestamp - timedelta(seconds=duration)
            after_event_timestamp = event_timestamp + timedelta(seconds=duration)

            start_file_index = file_time_df.index.get_loc(pre_event_timestamp,
                                                          method="ffill")  # use the pre-event timestamp, damit man das fenster bekommt
            events_df.iloc(row_index)["StartFileIndex"] = start_file_index
            events_df.iloc(row_index)["StartFileFilename"] = file_time_df.iloc(start_file_index)["Filepath"]

            # LOAD the start_file
            start_file_df, start_file_info = self.load_File(file_time_df.iloc(start_file_index)["Filepath"])

            # Check if file überschreitend
            if after_event_timestamp > file_time_df.iloc(start_file_index)[str("Current " + phase + " End")]:
                start_file_index_sligthly_higher = start_file_index + timedelta(seconds=1)

                end_file_index = file_time_df.index.get_loc(pre_event_timestamp,
                                                            method="backfill")  # the next higher one
                events_df.iloc(row_index)["EndFileIndex"] = end_file_index
                events_df.iloc(row_index)["EndFileFilename"] = file_time_df.iloc(end_file_index)["Filepath"]

                end_file_df, end_file_info = self.load_File(file_time_df.iloc(end_file_index)["Filepath"])

                start_file_df.append(end_file_df)

            # Dann den Pre-Event Timestamp finden
            start_time_difference = pre_event_timestamp - start_file_info[str("current_" + phase)]["date_time"]
            start_offset = int(
                start_time_difference.total_seconds() / start_file_info[str("current_" + phase)]["measurement_steps"])

            # start time offset + 2 * duration
            end_time_difference = (pre_event_timestamp + (2 * timedelta(seconds=duration)))
            end_offset = (end_time_difference.total_seconds() / start_file_info[str("current_" + phase)][
                "measurement_steps"])

            # TODO SLice the rows in the dataframe --> save the signal to disk and name it appropriately

            event_file_name = start_file_info

            # TODO for each row find and cut the event

    def load_Metadata(self, file_path):
        pass

    def load_Batch(self):
        pass

    def save_File(self, filepath, voltage, current):
        pass

    def calibrate_Sample(self, voltage, current):
        pass

class Electrical_Metrics:
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

        Args:
            instant_voltage: numpy array
            instant_current: numpy array

        Returns:
            active power: numpy array
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
        S = Vrms * Irms
        Args:
            instant_voltage: numpy array
            instant_current: numpy array
            period_length: numpy array

        Returns:
            apparent power: numpy array
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

        Q = sqrt(S^2 - P^2)
        Args:
            apparent_power: numpy array
            active_power: numpy array

        Returns:
            reactive power: numpy array

        """
        if len(apparent_power) == len(active_power):
            reactive_power = np.sqrt((apparent_power * apparent_power) - (active_power * active_power))
            return reactive_power
        else:
            raise ValueError("Signals need to have the same length")

    def compute_power_factor(self,apparent_power,active_power):
        """
        PF = P / S
        Args:
            apparent_power: numpy array
            active_power: numpy array

        Returns:
            power factor: single integer
        """

        power_factor = active_power / apparent_power
        return power_factor

    def compute_voltage_current_rms(self, voltage, current, period_length):
        """

        Args:
            voltage: numpy array
            current: numpy array
            period_length: integer

        Returns:
            voltage_rms: numpy array
            current_rms: numpy array
        """
        period_length = int(period_length)
        voltage_rms = self.compute_single_rms(voltage, period_length)
        current_rms = self.compute_single_rms(current, period_length)
        return voltage_rms, current_rms

    def compute_single_rms(self,signal,period_length):
        """

        Args:
            signal: numpy array
            period_length: in samples: can be the net frequency or a multiple of it

        Returns:
            rms_values
        """
        rms_values = []
        period_length = int(period_length)
        for i in range(0, len(signal), period_length):
            if i + period_length <= len(signal):
                signal_one_period = signal[i:int(i + period_length)]
                rms_one_period = np.sqrt(np.mean(np.square(signal_one_period))) #rms
                rms_values.append(rms_one_period)
        return np.array(rms_values)

