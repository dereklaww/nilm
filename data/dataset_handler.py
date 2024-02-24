import time
import os
from typing import List, Tuple

import plotly.express as px
import numpy as np
from fuzzywuzzy import fuzz
from nilmtk import DataSet, MeterGroup
import pandas as pd

# from nilmlab.lab_exceptions import LabelNormalizationError
from utils.logger import timing, TIMING, info, debug

dirname = os.path.dirname(__file__)
UK_DALE = os.path.join(dirname, r'Datasets\\ukdale.h5')


NAME_UK_DALE = 'UK DALE'
SITE_METER = 'Site meter'

class NoSiteMeterException(Exception):
    pass

class LabelNormalizationError(Exception):
    pass

class DataSetGenerator():

    def __init__(self, dataset: DataSet, name: str):
        self.dataset = dataset
        self.name = name

    def get_dataset(self):
        return self.dataset

    def get_name(self):
        return self.name
    
    def read_all_meters(self, data_window: Tuple[str, str], sample_period: int = 6, building: int = 1) -> Tuple[pd.DataFrame, MeterGroup]:
        '''
        Retrieve all meter data within data window from the dataset.
        Args:
            data_window (str, str): in the format of 'YYYY-MM-DD'
            sample_period (int): in seconds
            building (int): The building to read the records from.
        Returns:
            Dataframe and MeterGroup
        '''

        start_time = time.time() if TIMING else None
        self.dataset.set_window(start=data_window[0], end=data_window[1])
        elec = self.dataset.buildings[building].elec
        timing('NILMTK selecting all meters: {}'.format(round(time.time() - start_time, 2)))

        start_time = time.time() if TIMING else None
        df = elec.dataframe_of_meters(sample_period=sample_period)
        timing('NILMTK converting all meters to dataframe: {}'.format(round(time.time() - start_time, 2)))

        df.fillna(0, inplace=True)
        return df, elec
    
    def read_selected_appliances(self, appliances: List, data_window = Tuple[str, str], sample_period=6, building=1,
                                 include_mains=True) -> Tuple[pd.DataFrame, MeterGroup]:
        """
        Retrieve all meter data of selected appliances within data window from the dataset.
        Args:
            appliances (List): A list of appliances to read their records.
            include_mains (bool): True if should include main meters.
            data_window (str, str): in the format of 'YYYY-MM-DD'
            sample_period (int): in seconds
            building (int): The building to read the records from.

        Returns:
            Dataframe and MeterGroup
        """

        debug(f" read_selected_appliances {appliances}, {building}, {data_window[0]} - {data_window[1]}, include mains:{include_mains}")

        selected_metergroup = self.get_selected_metergroup(appliances, data_window, building = building, sample_period=sample_period, include_mains=include_mains)

        start_time = time.time() if TIMING else None
        df = selected_metergroup.dataframe_of_meters(sample_period=sample_period)
        timing('NILMTK converting specified appliances to dataframe: {}'.format(round(time.time() - start_time, 2)))

        debug(f"Length of data of read_selected_appliances {len(df)}")
        df.fillna(0, inplace=True)
        return df, selected_metergroup

    def get_selected_metergroup(self, appliances: List, data_window = Tuple[str, str], sample_period=6, building=1,
                                 include_mains=True) -> MeterGroup:
        """
        Retrieve all meter data of selected appliances within data window from the dataset.
        Args:
            appliances (List): A list of appliances to read their records.
            building (int): The building to read the records from.
            data_window (str, str): in the format of 'YYYY-MM-DD'
            include_mains (bool): True if should include main meters.

        Returns:
            A MeterGroup containing the specified appliances.
        """
        start_time = time.time() if TIMING else None
        self.dataset.set_window(start=data_window[0], end=data_window[1])
        elec = self.dataset.buildings[building].elec

        appliances_singular_meter = []
        appliances_multiple_meters = []

        # filter appliances with more than one elec meter
        for appliance in appliances:
            metergroup = elec.select_using_appliances(type=appliances)
            if len(metergroup.meters) > 1:
                appliances_multiple_meters.append(appliance)
            else:
                appliances_singular_meter.append(appliance)

        # instance = instance of appliance in the building
        special_metergroup = None
        for appliance in appliances_multiple_meters:
            inst = 1
            if appliance == 'sockets' and building == 3:
                inst = 4
            if special_metergroup is None:
                special_metergroup = elec.select_using_appliances(type=appliance, instance=inst)
            else:
                special_metergroup = special_metergroup.union(elec.select_using_appliances(type=appliance, instance=1))

        selected_metergroup = elec.select_using_appliances(type=appliances_singular_meter)
        selected_metergroup = selected_metergroup.union(special_metergroup)
        if include_mains:
            mains_meter = self.dataset.buildings[building].elec.mains()
            if isinstance(mains_meter, MeterGroup):
                if len(mains_meter.meters) > 1:
                    mains_meter = mains_meter.meters[0]
                    mains_metergroup = MeterGroup(meters=[mains_meter])
                else:
                    mains_metergroup = mains_meter
            else:
                mains_metergroup = MeterGroup(meters=[mains_meter])
            selected_metergroup = selected_metergroup.union(mains_metergroup)
        timing('NILMTK select using appliances: {}'.format(round(time.time() - start_time, 2)))
        return selected_metergroup
    
    def read_mains(self, data_window:Tuple[str, str] , sample_period=6, building=1) -> Tuple[pd.DataFrame, MeterGroup]:
        """
        Loads the data of the specified appliances.
        Args:
            sample_period (int): The sample period of the records.
            building (int): The building to read the records from.

        Returns:
            DataFrame and MeterGroup of the data that are read.
        """
        self.dataset.set_window(start=data_window[0], end=data_window[1])
        mains_meter = self.dataset.buildings[building].elec.mains()
        if isinstance(mains_meter, MeterGroup):
            mains_metergroup = mains_meter
        else:
            mains_metergroup = MeterGroup(meters=[mains_meter])
        start_time = time.time() if TIMING else None
        df = mains_metergroup.dataframe_of_meters(sample_period=sample_period)
        timing('NILMTK converting mains to dataframe: {}'.format(round(time.time() - start_time, 2)))

        df.fillna(0, inplace=True)
        return df, mains_metergroup

    @staticmethod
    def normalize_columns(df: pd.DataFrame, meter_group: MeterGroup, appliance_names: List[str]) -> Tuple[pd.DataFrame, dict]:
        """
        It normalizes the names of the columns for compatibility.
        Args:
            df (DataFrame):
            meter_group (MeterGroup):
            appliance_names (List[str]):

        Returns:
            A tuple with a DataFrame and a dictionary mapping labels to ids.
        """
        labels = meter_group.get_labels(df.columns)
        label_counts = {}
        normalized_labels = []
        info(f"Df columns before normalization {df.columns}")
        info(f"Labels before normalization {labels}")

        for label in labels:
            if label == SITE_METER and SITE_METER not in appliance_names:
                normalized_labels.append(SITE_METER)
                continue
            if label not in label_counts:
                label_counts[label] = 0
                normalized_labels.append(f"{label}_0")
            else:
                label_counts[label] += 1
                normalized_labels.append(f"{label}_{label_counts[label]}")

        if len(normalized_labels) != len(labels):
            debug(f"len(normalized_labels) {len(normalized_labels)} != len(labels) {len(labels)}")
            raise LabelNormalizationError()
        label2id = {l: i for l, i in zip(normalized_labels, df.columns)}
        df.columns = normalized_labels
        info(f"Normalized labels {normalized_labels}")
        return df, label2id
    
class DataSetHandler():
   @staticmethod
   def create_uk_dale_dataset():
       return DataSetGenerator(DataSet(UK_DALE), NAME_UK_DALE)
   
def plot_sequence(sequence: pd.DataFrame, plot=False, save_figure=False, filename=None):
    if plot or save_figure:
        fig = px.line(sequence)
        if filename is not None and save_figure:
            fig.write_image(filename + '.png')
        if plot:
            px.show()
