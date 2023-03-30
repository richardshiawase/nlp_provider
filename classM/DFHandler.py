import pickle
import re
from multiprocessing import Pool
import django

from classM.ColumnToRead import ColumnToRead
from classM.MasterData import MasterData
from classM.PerbandinganResult import PerbandinganResult
from classM.PredictionId import PredictionId

django.setup()
import pandas as pd
from tqdm import tqdm

import warnings

from classM.ExcelBacaTulis import ExcelBacaTulis
from classM.ItemProvider import ItemProvider
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
        print("Cacah Dataframe")
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
            # df_list.append([df_new,lr])
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

    def process_result_file_to_dataframe(self, process_dict):
        print("Process result file to dataframe")
        pd.options.display.max_colwidth = None
        provider_name_list = []
        score_list = []
        provider_object_list = []
        provider_name_predict_list = []
        df_dataset_non_duplicate = self.df_dataset
        alamat_prediction_list = []
        for key, value_list in process_dict.items():
            if (key == "nama_alamat"):

                for value in value_list:
                    # search the provider name among the course_title value
                    # val = (df_non_duplicate['course_title'].eq(value))
                    # res = df_non_duplicate[val]
                    try:
                        provider_name_list.append(value)

                        # predict the text !
                        sample1 = self.vectorize_text(value, self.tfidf_vec1)
                        y_preds = self.loaded_model1.predict(sample1)

                        # add prediction ke list
                        y_preds = str(y_preds).replace("[", "").replace("]", "").replace("'", "")
                        provider_name_predict_list.append(y_preds)

                        # calculate proba
                        p = self.loaded_model1.predict_proba(sample1)
                        ix = p.argmax(1).item()
                        nil = (f'{p[0, ix]:.2}')

                        # add proba to list
                        score_list.append(nil)
                        # jika prediksi sama dengan dataset maka ambil alamat dari dataset
                        val_master = (df_dataset_non_duplicate['subject'].eq(y_preds))
                        try:
                            res_master = df_dataset_non_duplicate[val_master]
                            al = res_master["alamat"].head(1)
                            alamat_prediction_list.append(al.values[0])
                        except:
                            try:
                                bool_find_in_dataset = (df_dataset_non_duplicate['subject'].str.contains(re.escape(y_preds)))
                                res_master = df_dataset_non_duplicate[bool_find_in_dataset]
                                al = res_master["alamat"].head(0)
                                alamat_prediction_list.append(al.values[0])
                            except Exception as e:
                                print(str(e))
                                alamat_prediction_list.append("-")
                    except:
                        print("err")

        # print(process_dict["Nama"])
        # print(process_dict["Alamat"])

        process_dict["Prediction"] = pd.Series(provider_name_predict_list)
        process_dict["Score"] = pd.Series(score_list)
        process_dict["Alamat_Prediction"] = pd.Series(alamat_prediction_list)
        result_df = pd.DataFrame(process_dict)

        return result_df




    def pool_handler(self, dataframe):
        concatenated_dict = {}
        key_list = []
        # # # Split dataframe to many
        df_list = self.cacah_dataframe(dataframe)

        # # # Using multiprocess with pool as many as dataframe list
        p = Pool(len(df_list))

        # # # Use Pool Multiprocessing
        processed_list = p.map(self.pool_process_pure, df_list)

        for list in processed_list:
            # add array values to this key
            # create array relative to key
            # if key == desired_key then append the key
            for key, value in list.items():
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
        # set header if any
        if header is not None:
            self.column.set_column_to_read_when_process(header)


        # mapping the column and content
        data = self.create_dataframe_with_favourable_column(dataframe_pembanding)

        # start multi process calculation
        processed_dict = self.pool_handler(data)

        return processed_dict

    def set_perbandingan_model(self, perbandingan_model_obj):
        self.perbandingan_model = perbandingan_model_obj

    def add_to_provider_list(self):
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

                provider_object = ItemProvider(provider_name, alamat, y_preds, nil, 0, 0, 0)
                provider_object.set_selected(compared)
                provider_object.set_alamat_prediction(alamat_prediction)
                self.provider_list.append(provider_object)

    def get_provider_list(self):
        return self.provider_list

    def get_provider_list_json_response(self):
        ls = []
        for obj in self.provider_list:
            ls.append(obj.__dict__)
        return ls

    def process_result_id_master_to_dataframe(self, processed_dict):
        print("\n\nCreate Result File With ID")
        # read excel master provider and get the dataframe
        id_master_list = []
        alamat_master_list = []
        provider_name_master_list = []
        master_dict = {}

        df_result = self.get_pembanding_result_dataframe()

        # # # # # # # # # # # # # # # PREPARE MASTER DATA TO DICT
        for key, value in processed_dict.items():

            if key == "ProviderId":
                for data in value:
                    master_data = MasterData()
                    master_data.set_id_master(data)
                    master_dict[data]=master_data

        for key, value in processed_dict.items():
            if key == "PROVIDER_NAME":
                for data in value:
                    for k,v in master_dict.items():
                            if v.get_match_name_boolean() is False:
                                v.set_nama_master(data)
                                v.set_match_name_true()
                                break

        for key, value in processed_dict.items():
            if key == "ADDRESS":
                for data in value:
                    for k,v in master_dict.items():
                        if v.get_match_address_boolean() is False:
                            v.set_alamat_master(data)
                            v.set_match_address_true()
                            break

        # # # # # # # # # # # # # # # # # # # #



        for prediction in df_result['Prediction']:
            for master_obj in master_dict.values():
                if prediction == master_obj.get_nama_master():
                    id_master_list.append(master_obj.get_id_master())
                    provider_name_master_list.append(master_obj.get_nama_master())
                    alamat_master_list.append(master_obj.get_alamat_master())
                    break
            if prediction not in provider_name_master_list:
                provider_name_master_list.append("-")
                id_master_list.append("-")
                alamat_master_list.append("-")

        dict_result = {
            'IdMaster': pd.Series(id_master_list),
            'Master_Nama': pd.Series(provider_name_master_list),
            'Master_Alamat': pd.Series(alamat_master_list),
            'Nama': pd.Series(df_result['Nama']),
            'Alamat': pd.Series(df_result['Alamat']),
            'Score' : pd.Series(df_result['Score']),
            'RI' :  pd.Series(df_result['RI']),
            'RJ' : pd.Series(df_result['RJ'])
        }
        result_df = pd.DataFrame(dict_result)

        return result_df


