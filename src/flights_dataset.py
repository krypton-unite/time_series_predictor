"""
FlightsDataset
"""

import calendar
import math
import numpy as np
import pandas as pd
import seaborn as sns

from time_series_predictor import TimeSeriesDataset


def _raw_make_predictor(features, *reshape_args):
    #pylint: disable=too-many-function-args
    return np.concatenate(features, axis=-1).reshape(
        list(reshape_args) + [len(features)]).astype(np.float32)


def _make_predictor(features, number_of_training_examples):
    #pylint: disable=too-many-function-args
    return _raw_make_predictor(features, number_of_training_examples, -1)


def _get_labels(input_features, output_features):
    def _features_to_label_list(features):
        return [list(feature)[0] for feature in features]
    labels = {}
    labels['x'] = _features_to_label_list(input_features)
    labels['y'] = _features_to_label_list(output_features)
    return labels


class FlightsDataset(TimeSeriesDataset):
    """
    FlightsDataset class

    :param except_last_n: initialize the FlightsDataset without n last months
    """

    # pylint: disable=too-many-locals
    def __init__(self, except_last_n=0):
        flights_dataset = sns.load_dataset("flights")
        chopped_flights_dataset = flights_dataset[:len(flights_dataset)-except_last_n]
        passengers = chopped_flights_dataset['passengers']
        month = chopped_flights_dataset['month']
        year = chopped_flights_dataset['year']

        month_number = [list(calendar.month_name).index(_month)
                        for _month in month]

        passengers_df = pd.DataFrame(passengers)
        month_number_df = pd.DataFrame(data={'month_number': month_number})
        year_df = pd.DataFrame(year)

        number_of_training_examples = 1
        # Store month_number and year as _x
        input_features = [month_number_df, year_df]
        _x = _make_predictor(input_features, number_of_training_examples)

        # Store passengers as _y
        output_features = [passengers_df]
        _y = _make_predictor(output_features, number_of_training_examples)

        super().__init__(_x, _y, _get_labels(input_features, output_features))
        self.month_number_df = month_number_df
        self.year_df = year_df

    # pylint: disable=arguments-differ
    def make_future_dataframe(self, scaler, number_of_months, include_history=True):
        """
        make_future_dataframe

        :param number_of_months: number of months to predict ahead
        :param include_history: optional, selects if training history is to be included or not
        :returns: future dataframe with the selected amount of months
        """
        def create_dataframe(name, data):
            return pd.DataFrame(data={name: data})

        def create_month_dataframe(data):
            return create_dataframe('month_number', data)

        def create_year_dataframe(data):
            return create_dataframe('year', data)

        month_number_df = self.month_number_df
        year_df = self.year_df
        last_month = month_number_df.values[-1][0]
        last_year = year_df.values[-1][0]
        if not include_history:
            month_number_df = create_month_dataframe([])
            year_df = create_year_dataframe([])
        for i in range(number_of_months):
            month_index = last_month+i
            new_months = [math.fmod(month_index, 12)+1]
            new_years = [last_year + math.floor(month_index / 12)]
            month_number_df = month_number_df.append(
                create_month_dataframe(new_months), ignore_index=True)
            year_df = year_df.append(
                create_year_dataframe(new_years), ignore_index=True)
        input_features = [month_number_df, year_df]
        return scaler.two_d_transform(_raw_make_predictor(input_features, -1))
