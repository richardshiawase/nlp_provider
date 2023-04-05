import pickle
import re
from multiprocessing import Pool
import django
from django.forms import model_to_dict

from classM.ColumnToRead import ColumnToRead
from classM.MasterData import MasterData

django.setup()
import pandas as pd

import warnings

from classM.ExcelBacaTulis import ExcelBacaTulis
from model.models import ItemProvider, Provider
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
        self.master_provider = MasterData()

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


    def error_value_write(self,item_provider,e):
        item_provider.set_label_name("-")
        item_provider.set_proba_score(0)
        item_provider.set_count_label_name(0)
        item_provider.set_alamat_prediction("-")

        # item_provider.save()
        print("Error read the file "+item_provider.get_nama_provider() +" "+ str(e))
    def comparing_item_provider_to_ml_and_save_item(self):
        print("Process result file to dataframe")
        pd.options.display.max_colwidth = None
        df_dataset_non_duplicate = self.df_dataset
        alamat_prediction_list = []
        for item_provider in self.perbandingan_model.get_list_item_provider():
            try:
                # predict the text !
                nama_alamat = item_provider.get_nama_alamat()
                sample1 = self.vectorize_text(nama_alamat, self.tfidf_vec1)
                y_preds = self.loaded_model1.predict(sample1)

                # add prediction ke list
                y_preds = str(y_preds).replace("[", "").replace("]", "").replace("'", "")

                # calculate proba
                p = self.loaded_model1.predict_proba(sample1)
                ix = p.argmax(1).item()
                nil = (f'{p[0, ix]:.2}')

                # add proba to list
                # jika prediksi sama dengan dataset maka ambil alamat dari dataset
                val_master = (df_dataset_non_duplicate['subject'].eq(y_preds))
                try:
                    res_master = df_dataset_non_duplicate[val_master]
                    al = res_master["alamat"].head(1)
                    item_provider.set_label_name(y_preds)
                    item_provider.set_proba_score(nil)
                    item_provider.set_count_label_name(0)
                    item_provider.set_alamat_prediction(al.values[0])
                    # item_provider.save()
                except:
                    try:
                        bool_find_in_dataset = (
                            df_dataset_non_duplicate['subject'].str.contains(re.escape(y_preds)))
                        res_master = df_dataset_non_duplicate[bool_find_in_dataset]
                        al = res_master["alamat"].head(0)

                        item_provider.set_label_name(y_preds)
                        item_provider.set_proba_score(nil)
                        item_provider.set_count_label_name(0)
                        item_provider.set_alamat_prediction(al.values[0])
                        # item_provider.save()

                    except Exception as e:
                        self.error_value_write(item_provider,e)


            except Exception as e:
                self.error_value_write(item_provider,e)



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

    def add_to_provider_list(self,file_location):
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
                item_provider.set_proba_score(nil)
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

    def process_result_id_master_to_dataframe(self):
        print("\n\nCreate Result File With ID")
        # read excel master provider and get the dataframe
        id_master_list = []
        alamat_master_list = []
        provider_name_master_list = []
        list_item_provider_nama = []
        list_item_provider_alamat = []
        list_item_provider_score = []
        list_item_provider_ri = []
        list_item_provider_rj = []

        df_result_2 = self.perbandingan_model.get_list_item_provider()


        for item in df_result_2:
            for master_obj in self.master_provider.get_list_item_master_provider():
                if item.get_label_name() == master_obj.get_nama_master():
                    id_master_list.append(master_obj.get_id_master())
                    provider_name_master_list.append(master_obj.get_nama_master())
                    alamat_master_list.append(master_obj.get_alamat_master())
                    list_item_provider_nama.append(item.get_nama_provider())
                    list_item_provider_alamat.append(item.get_alamat())
                    list_item_provider_ri.append(item.get_ri())
                    list_item_provider_rj.append(item.get_rj())
                    list_item_provider_score.append(item.get_proba_score())
                    break
            if item.get_label_name() not in provider_name_master_list:
                provider_name_master_list.append("-")
                id_master_list.append("-")
                alamat_master_list.append("-")
                list_item_provider_nama.append(item.get_nama_provider())
                list_item_provider_alamat.append(item.get_alamat())
                list_item_provider_ri.append("-")
                list_item_provider_rj.append("-")
                list_item_provider_score.append(0)


        dict_result = {
            'IdMaster': pd.Series(id_master_list),
            'Master_Nama': pd.Series(provider_name_master_list),
            'Master_Alamat': pd.Series(alamat_master_list),
            'Nama': pd.Series(list_item_provider_nama),
            'Alamat': pd.Series(list_item_provider_alamat),
            'Score': pd.Series(list_item_provider_score),
            'RI': pd.Series(list_item_provider_ri),
            'RJ': pd.Series(list_item_provider_rj)
        }
        result_df = pd.DataFrame(dict_result)
        return result_df

    def create_master_provider_item_list(self, dataframe_pembanding):
        # get specified column to read
        print("Create provider item list")
        master_item_list = []
        for row in dataframe_pembanding.itertuples(index=True, name='Sheet1'):

            master_data = MasterData()

            master_provider_id = row.ProviderId
            master_state_id = row.stateId
            master_city_id = row.cityId
            master_category_1 = row.Category_1
            master_category_2 = row.Category_2
            master_nama_provider = row.PROVIDER_NAME
            master_alamat = row.ADDRESS
            master_tel = row.TEL_NO



            master_data.set_id_master(master_provider_id)
            master_data.set_nama_master(master_nama_provider)
            master_data.set_alamat_master(master_alamat)
            master_data.set_state_id_master(master_state_id)
            master_data.set_city_id_master(master_city_id)
            master_data.set_category_1_master(master_category_1)
            master_data.set_category_2_master(master_category_2)
            master_data.set_telepon_master(master_tel)


            master_item_list.append(master_data)

        self.master_provider.set_list_item_master_provider(master_item_list)

    def create_provider_item_list(self, dataframe_pembanding):
        # get specified column to read
        print("Create provider item list")
        provider_item_list = []
        for row in dataframe_pembanding.itertuples(index=True, name='Sheet1'):
            provider_object = ItemProvider()
            provider_object.set_id_asuransi(self.perbandingan_model.get_id_asuransi_model())

            nama = row.Nama
            alamat = row.Alamat
            rawat_inap = row.RI
            rawat_jalan = row.RJ

            provider_object.set_provider_name(nama)
            provider_object.set_alamat(alamat)
            provider_object.set_ri(rawat_inap)
            provider_object.set_rj(rawat_jalan)
            provider_object.set_label_name("-")
            provider_object.set_proba_score(0)
            provider_object.set_count_label_name(0)
            provider_object.set_nama_alamat()

            provider_item_list.append(provider_object)

        self.perbandingan_model.set_list_item_provider(provider_item_list)

