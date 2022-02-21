import pandas as pd

import src.data.make_dataset as md
import src.data.solar_radiation as sr
import src.data.weather_data as wd


import src.Datahandler.prepare_data as prep


filepath = "data/raw/elspotprices_7.2.2022.dk1.csv"
createNewPickle = False


makeWeahterData = False
makeSolarradiationData = True


def main():
    if makeSolarradiationData:
        solar_radiation = sr.GetSolarRadiationData()
    if makeWeahterData:
        weatherdata = wd.GetWeatherData()
    if not makeWeahterData:
        weatherdata = pd.read_pickle("weatherdata_2013_2019_edmonton.pkl")
    print("hi")
    # if createNewPickle:
    #     data = md.clean_data(filepath)
    # else:
    #     data = pd.read_pickle("data/processed/cleaned_data.pkl")
    #
    # train_scaled, validation_scaled, test_scaled = prep.data_prep(data)
    # print("hi")
    #

if __name__ == '__main__':
    main()
