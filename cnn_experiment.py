import traceback

from env.exp_config import GenericExperiment
from data.dataset_handler import DataSetHandler
from env.env_config import EnvironmentHandler, TimeSeriesLength
from utils.logger import debug


APPLIANCES_UK_DALE_BUILDING_1 = ['oven', 'microwave', 'dish washer', 'fridge freezer', 'kettle', 'washer dryer',
                'toaster', 'boiler', 'television', 'hair dryer', 'vacuum cleaner', 'light']

ukdale_train_year_start = '2013'
ukdale_train_year_end = '2013'
ukdale_train_month_end = '5'
ukdale_train_month_start = '3'
ukdale_train_end_date = "{}-30-{}".format(ukdale_train_month_end, ukdale_train_year_end)
ukdale_train_start_date = "{}-1-{}".format(ukdale_train_month_start, ukdale_train_year_start)
ukdale_train_date_window = (ukdale_train_start_date, ukdale_train_end_date)

ukdale_test_year_start = '2014'
ukdale_test_year_end = '2014'
ukdale_test_month_end = '12'
ukdale_test_month_start = '6'
ukdale_test_end_date = "{}-30-{}".format(ukdale_test_month_end, ukdale_test_year_end)
ukdale_test_start_date = "{}-1-{}".format(ukdale_test_month_start, ukdale_test_year_start)
ukdale_test_date_window = (ukdale_test_start_date, ukdale_test_end_date)

env_ukdale_building_1 = EnvironmentHandler.create_env_single_building(
    datasource=DataSetHandler.create_uk_dale_dataset(),
    building=1,
    sample_period=6,
    train_date_window=ukdale_train_date_window,
    test_date_window=ukdale_test_date_window,
    appliances=APPLIANCES_UK_DALE_BUILDING_1)

ukdale_building1_experiment = GenericExperiment(env_ukdale_building_1)

def run_experiments(experiment, appliances, window):
    experiment.setup_running_params(
        train_appliances=appliances,
        test_appliances=appliances,
        ts_len=window,
        repeat=1)
    tb = "No error"
    try:
        experiment.run()
    except Exception as e:
        tb = traceback.format_exc()
        debug(tb)
        debug(f"{e}")

if __name__ == "__main__":
    run_experiments(ukdale_building1_experiment, APPLIANCES_UK_DALE_BUILDING_1, TimeSeriesLength.WINDOW_4_HOURS)

