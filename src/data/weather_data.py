import pandas as pd
import requests
from io import StringIO
import numpy as np
from datetime import datetime, timedelta



def GetWeatherData():
    start_date = datetime(2013, 1, 1, 00, 00, 00)
    end_date = datetime(2019, 1, 1, 00, 00, 00)
    temp_date = datetime(2013, 1, 21, 00, 00, 00)
    location = "groningen, netherlands"

    url = "https://visual-crossing-weather.p.rapidapi.com/history"


    # querystring = {"startDateTime": "2018-12-12T00:00:00", "aggregateHours": "1", "location": "groningen, netherlands",
    #                "endDateTime": "2019-01-01T00:00:00", "unitGroup": "metric", "dayStartTime": "8:00:00",
    #                "contentType": "csv", "shortColumnNames": "0"}

    headers = {
        'x-rapidapi-host': "visual-crossing-weather.p.rapidapi.com",
        'x-rapidapi-key': "c6e51148a1msh1f6168357a6bce4p1dd6f8jsnd2b05854b27d"
    }
    responses = pd.DataFrame()
    while start_date < end_date:
        querystring = {"startDateTime": start_date.strftime("%Y-%m-%dT%H:%M:%S"), "aggregateHours": "1",
                       "location": location,
                       "endDateTime": temp_date.strftime("%Y-%m-%dT%H:%M:%S"), "unitGroup": "metric",
                       "dayStartTime": "8:00:00",
                       "contentType": "csv", "shortColumnNames": "0"}
        response = requests.request("GET", url, headers=headers, params=querystring)
        s = str(response.content, 'utf-8')
        data_content = StringIO(s)
        df = pd.read_csv(data_content)
        responses = responses.append(df, ignore_index=True)
        temp_date = temp_date + timedelta(days=21)
        start_date = start_date + timedelta(days=21)

    responses.to_pickle("./weatherdata_2013_2019_edmonton.pkl")
    print("response")
    return response