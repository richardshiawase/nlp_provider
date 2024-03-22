import pickle
import re
from multiprocessing import Pool
import django
from django.forms import model_to_dict
from fuzzywuzzy import fuzz

from classM.ColumnToRead import ColumnToRead

import pandas as pd

import warnings

from classM.ExcelBacaTulis import ExcelBacaTulis

from classM.Pembersih import Pembersih

warnings.simplefilter(action='ignore', category=FutureWarning)


class DFHandler:
    def __init__(self):
        self.provider_name_predict_list = []
        filename = 'tfidf_vec.pickle'
        self.tfidf_vec1 = pickle.load(open(filename, 'rb'))
        filename = 'finalized_model.sav'
        self.loaded_model1 = pickle.load(open(filename, 'rb'))
        self.ex = ExcelBacaTulis()
        self.provider_list = []
        self.df = None
        self.column = ColumnToRead()
        # self.master_provider = MasterData()

    def convert_to_dataframe_from_excel(self, excel):
        self.df = self.ex.baca_excel(excel)

        # clean the dataframe
        self.pembersih = Pembersih(self.df)

        df = self.pembersih._return_df()
        return df

    def set_dataframe(self, dataframe):
        if dataframe is not None:
            self.pembersih = Pembersih(dataframe)
            self.dataframe = self.pembersih._return_df()

    def get_data_frame(self):
        df = self.pembersih._return_df()
        return df

    def get_processed_result(self):
        ts = pd.concat(list(self.concatenated_dataframe), ignore_index=True)
        return ts

    def cacah_dataframe(self, df):
        print("Cacah Dataframes")
        split_row_each = 800
        start_index = 0
        iteration_count = int(df.shape[0] / split_row_each)
        sisa = df.shape[0] % split_row_each
        sisa_row = iteration_count * split_row_each + sisa
        df_list = []
        for x in range(iteration_count):
            end_index = start_index + split_row_each
            df_new = df.iloc[start_index:end_index]
            start_index = end_index
            df_list.append(df_new)
        aw = lambda x, y: y if x > 0 else 0
        df_last = df.iloc[start_index:aw(sisa, sisa_row)]
        df_list.append(df_last)

        return df_list

    def set_df_dataset(self, df):
        self.df_dataset = df

    def vectorize_text(self, text, tfidf_vec):
        # text = "Klinik Ananda"
        my_vec = tfidf_vec.transform([text])
        return my_vec.toarray()

    def pool_process_pure(self, df_list):
        print("Multi Process Dataframe")
        # for df in df_list:
        # dataframe Name and Age columns
        pd.options.display.max_colwidth = None
        process_dict = {}
        # map value of processed dataframe to process_df column
        for column in df_list:
            process_dict[column] = df_list[column].values.tolist()

        return process_dict



    def pool_handler(self, dataframe):
        concatenated_dict = {}
        key_list = []
        # # # Split dataframe to many
        df_list = self.cacah_dataframe(dataframe)

        # # # Using multiprocess with pool as many as dataframe list
        p = Pool(len(df_list))

        # # # Use Pool Multiprocessing
        processed_list = p.map(self.pool_process_pure, df_list)

        for plist in processed_list:
            # add array values to this key
            # create array relative to key
            # if key == desired_key then append the key
            for key, value in plist.items():
                if key not in key_list:
                    key_list.append(key)
                    concatenated_dict[key] = value
                else:
                    for data in value:
                        concatenated_dict.get(key).append(data)
        return concatenated_dict

    def create_dataframe_with_favourable_column(self, dataframe):
        # get specified column to read
        print("Mapping Content With Column")
        header = self.column.get_column_to_read_when_process()
        mapped = {}
        for x in header:
            try:
                mapped[x] = dataframe[x]
            except:
                print("Tidak ditemukan header " + str(x))

        # additional query
        try:
            mapped["nama_alamat"] = mapped["Nama"].map(str) + '#' + mapped["Alamat"].map(str)
        except:
            print("nama_alamat tidak ditemukan")

        # convert mapped dictionary header
        data = pd.DataFrame(mapped)

        return data

    def get_pembanding_result_dataframe(self):
        excel_result = self.perbandingan_model.get_file_location_result()
        self.convert_to_dataframe_from_excel(excel_result)
        return self.get_data_frame()

    def convert_dataframe_refer_to_specified_column(self, header=None, dataframe_pembanding=None):
        print("Process file excel")
        print(list(dataframe_pembanding.columns))
        # set header if any
        if header is not None:
            self.column.set_column_to_read_when_process(header)

        # mapping the column and content
        # data = self.create_dataframe_with_favourable_column(dataframe_pembanding)

        # start multi process calculation
        # processed_dict = self.pool_handler(data)
        # processed_dict = []

        # return processed_dict

    def set_perbandingan_model(self, perbandingan_model_obj):
        self.perbandingan_model = perbandingan_model_obj

    def add_to_provider_list(self, file_location):
        print("Add to provider list")
        self.convert_to_dataframe_from_excel(file_location)
        self.provider_list.clear()
        if self.df is not None:
            for index, row in self.df.iterrows():
                try:
                    provider_name = row['Nama']
                    y_preds = row["Prediction"]
                    alamat = row["Alamat"]
                    alamat_prediction = row["Alamat_Prediction"]
                    nil = row["Score"]
                    compared = False
                except:
                    alamat_prediction = "-"
                    nil = 0
                    compared = False

                item_provider = ItemProvider()
                item_provider.set_provider_name(provider_name)
                item_provider.set_alamat_prediction(alamat_prediction)
                item_provider.set_alamat(alamat)
                item_provider.set_label_name(y_preds)

                item_provider.set_selected(compared)
                data = model_to_dict(item_provider)
                self.provider_list.append(data)

    def get_provider_list(self):
        return self.provider_list

    def get_provider_list_json_response(self):
        ls = []
        for obj in self.provider_list:
            ls.append(obj)
        return ls




