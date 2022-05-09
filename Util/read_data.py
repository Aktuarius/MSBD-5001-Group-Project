import pandas
import numpy

class commmon_util:
    def __init__(self):
        label_y = self.read_crop_label()
        data_x = self_read_processed_histogram()


    def read_processed_histogram(self):
        pass

    def read_crop_label() -> pd.DataFrame:
        label_path = str(
            Path(os.getcwd()).parent.parent.absolute()) + '/Yield_Data/all_country_crop_yield_tons_per_hectare.csv'
        df = pd.read_csv(label_path)
        df = df.set_index(['Country Name'])
        return df
