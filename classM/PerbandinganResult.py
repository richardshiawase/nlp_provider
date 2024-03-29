import pandas as pd
import requests

from classM.ColumnOutput import ColumnOutput
from classM.ExcelBacaTulis import ExcelBacaTulis
from django.db import connection


class PerbandinganResult():
    def __init__(self):
        self.column_output = ColumnOutput()
        self.ex = ExcelBacaTulis()
        self.result_link = None

        pass

    def set_link_result_with_id_master(self, result_link):
        self.result_link = result_link

    def get_link_result_with_id_master(self):
        return self.result_link

    def map_list_item_provider_to_column_output(self, provider_item_list):
        print("Map List Item To Column Output")
        result_column = self.column_output.get_column_to_output()
        mappeds = {}
        list_nama = []
        list_alamat = []
        list_prediction_name = []
        list_alamat_prediction = []
        list_score = []
        list_total_score = []
        list_alamat_ratio = []
        list_ratio = []
        list_ri = []
        list_rj = []

        for item in provider_item_list:
            list_nama.append(item.get_nama_provider())
            list_alamat.append(item.get_alamat())
            list_prediction_name.append(item.get_label_name())
            list_alamat_prediction.append(item.get_alamat_prediction())
            list_score.append(item.get_proba_score())
            list_ri.append(item.get_ri())
            list_rj.append(item.get_rj())
            list_total_score.append(item.get_total_score())
            list_alamat_ratio.append(item.get_alamat_ratio())
            list_ratio.append(item.get_ratio())

        for x in result_column:
            try:
                if x == "Nama":
                    mappeds[x] = list_nama
                if x == "Alamat":
                    mappeds[x] = list_alamat
                if x == "Prediction":
                    mappeds[x] = list_prediction_name
                if x == "Alamat_Prediction":
                    mappeds[x] = list_alamat_prediction
                if x == "Score":
                    mappeds[x] = list_score
                if x == "Ratio":
                    mappeds[x] = list_ratio
                if x == "Alamat_Ratio":
                    mappeds[x] = list_alamat_ratio
                if x == "Total_Score":
                    mappeds[x] = list_total_score
                if x == "RI":
                    mappeds[x] = list_ri
                if x == "RJ":
                    mappeds[x] = list_rj


            except Exception as e:
                print("Tidak ditemukan output column " + x + " di dataframe " + str(e))
                mappeds[x] = pd.Series([])

        return mappeds

    def map_list_master_item_provider_to_column_output(self, provider_item_list):
        print("Map List Master Item To Column Output")
        result_column = self.column_output.get_column_to_output()
        mappeds = {}
        list_master_nama = []
        list_master_alamat = []

        list_nama = []
        list_alamat = []
        list_prediction_name = []
        list_alamat_prediction = []
        list_score = []
        list_ri = []
        list_rj = []
        list_id_master = []

        for item in provider_item_list:
            list_nama.append(item.get_nama_provider())
            list_alamat.append(item.get_alamat())
            list_prediction_name.append(item.get_label_name())
            list_alamat_prediction.append(item.get_alamat_prediction())
            list_score.append(item.get_proba_score())
            list_ri.append(item.get_ri())
            list_rj.append(item.get_rj())

        for x in result_column:
            try:
                if x == "IdMaster":
                    mappeds[x] = list_id_master
                if x == "Master_Nama":
                    mappeds[x] = list_master_nama
                if x == "Master_Alamat":
                    mappeds[x] = list_master_alamat

                if x == "Nama":
                    mappeds[x] = list_nama

                if x == "Alamat":
                    mappeds[x] = list_alamat

                if x == "Prediction":
                    mappeds[x] = list_prediction_name
                if x == "Alamat_Prediction":
                    mappeds[x] = list_alamat_prediction
                if x == "Score":
                    mappeds[x] = list_score

                if x == "Compared":
                    mappeds[x] = []

                if x == "Clean":
                    mappeds[x] = []

                if x == "RI":
                    mappeds[x] = list_ri
                if x == "RJ":
                    mappeds[x] = list_rj

            except Exception as e:
                print("Tidak ditemukan output column " + x + " di dataframe " + str(e))
                mappeds[x] = pd.Series([])

        return mappeds


    def delete_provider_item_hospital_insurances_with_id_insurances(self, df_handler):
        id_asuransi = df_handler.perbandingan_model.get_id_asuransi()
        url = 'https://www.asateknologi.id/api/inshos-del'
        myobj = {'id_insurance': id_asuransi}
        try:
            x = requests.post(url, json=myobj)
        except Exception as e:
            print(str(e))

    def insert_into_end_point_andika_assistant_item_provider(self, df_handler):
        link = self.get_link_result_with_id_master()
        dataframe_insert = pd.read_excel(link)
        dataframe_insert_new = dataframe_insert.loc[dataframe_insert['Validity'] == True]
        id_asuransi = df_handler.perbandingan_model.get_id_asuransi()
        url = 'https://www.asateknologi.id/api/inshos'
        count = 0
        for index, row in dataframe_insert_new.iterrows():
            myobj = {'hospitalId': row['IdMaster'], 'insuranceId': id_asuransi, 'outpatient': row['RJ'],
                     'inpatient': row['RI']}
            try:
                x = requests.post(url, json=myobj)
                c+=1
            except Exception as e:
                print(str(e))
        print("Inserted "+str(count))
        pass

    def create_file_result_with_id_master(self, df_handler):
        # self.create_file(df_handler)

        lokasi_excel = "master_provider.xlsx"
        #
        # # read excel by lokasi excel and get the dataframe
        dataframe_pembanding = df_handler.convert_to_dataframe_from_excel(lokasi_excel)

        # # create process excel and convert to dict
        df_handler.create_master_provider_item_list(dataframe_pembanding)
        #
        # # # # # combine the result with id
        # processed_dataframe = df_handler.process_result_id_master_to_dataframe()

        # # # # # write to excel
        # self.ex.write_to_excel(df_handler.perbandingan_model.get_nama_asuransi_model(), "_result_final",
        #                        processed_dataframe)
        # #
        # self.set_link_result_with_id_master(
        #     "media/" + df_handler.perbandingan_model.get_nama_asuransi_model() + "_result_final.xlsx")

        pass
