import traceback

from env.exp_config import GenericExperiment
from data.dataset_handler import DataSetHandler
from env.env_config import EnvironmentHandler, TimeSeriesLength
from utils.logger import debug


APPLIANCES_UK_DALE_BUILDING_2 = ['fridge', 'microwave', 'active speaker', 'external hard disk', 'kettle', 
                                 'computer', 'running machine', 'dish washer', 'games console', 'toaster', 
                                 'modem', 'broadband router', 'computer monitor', 'laptop computer', 
                                 'washing machine', 'cooker', 'rice cooker']

ukdale_train_year_start = '2013'
ukdale_train_year_end = '2013'
ukdale_train_month_end = '12'
ukdale_train_month_start = '6'
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

env_ukdale = EnvironmentHandler.create_env_single_building(
    datasource=DataSetHandler.create_uk_dale_dataset(),
    building=2,
    sample_period=60,
    train_date_window=ukdale_train_date_window,
    test_date_window=ukdale_test_date_window,
    appliances=APPLIANCES_UK_DALE_BUILDING_2)

ukdale_experiment = GenericExperiment(env_ukdale)

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
    run_experiments(ukdale_experiment, APPLIANCES_UK_DALE_BUILDING_2, TimeSeriesLength.WINDOW_4_HOURS)

