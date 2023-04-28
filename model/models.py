import pickle
import re

import django
import pandas as pd
import requests
from django.db import models
from django.db import connections
from django.forms import model_to_dict
from fuzzywuzzy import fuzz

from classM.ColumnOutput import ColumnOutput
from classM.ColumnToRead import ColumnToRead
from classM.DFHandler import DFHandler
from classM.Dataset import Dataset
from classM.ExcelBacaTulis import ExcelBacaTulis
from classM.Pembersih import Pembersih


# Create your models here.
class List_Processed_Provider(models.Model):
    id_provider = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

    def add_provider(self, provider):
        self.provider_list.append(provider)

    def set_empty_provider_list(self):
        self.provider_list = []

    def get_provider_list(self):
        return self.provider_list

    def get_provider_list_json(self):
        ls = []
        for provider in self.get_provider_list():
            del provider._state
            ls.append(provider.__dict__)
        return ls

    def get_a_provider_from_id(self, pk):
        id_provider = pk
        for data in self.get_provider_list():
            if data.get_primary_key_provider() == str(id_provider):
                return data


class Provider_Model(models.Model):
    model_name = models.CharField(max_length=30)
    accuracy_score = models.DecimalField(max_digits=5, decimal_places=2)
    model_location = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)


class ItemProvider(models.Model):
    id_model = models.CharField(max_length=500)
    id_asuransi = models.CharField(max_length=500)
    nama_provider = models.CharField(max_length=500)
    alamat = models.CharField(max_length=500)
    label_name = models.CharField(max_length=300)
    proba_score = models.CharField(max_length=10)
    ratio = models.CharField(max_length=10)
    alamat_ratio = models.CharField(max_length=10)
    total_score = models.CharField(max_length=10)
    count_label_name = models.CharField(max_length=2)
    ri = models.CharField(max_length=2)
    rj = models.CharField(max_length=2)
    nama_alamat = models.CharField(max_length=500)
    alamat_prediction = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)
    golden_record = models.CharField(max_length=2)

    def set_compared(self, bool):
        self.compared = bool

    def get_compared(self):
        return self.compared

    def set_golden_record(self, bool):
        self.golden_record = bool

    def get_golden_record_status(self):
        return self.golden_record

    def set_id_master(self, id_master):
        self.id_master = id_master

    def get_id_master(self):
        return self.id_master

    def set_nama_master_provider(self, nama_master_provider):
        self.nama_master_provider = nama_master_provider

    def get_nama_master_provider(self):
        return self.nama_master_provider

    def set_alamat_master_provider(self, alamat_master):
        self.alamat_master_provider = alamat_master

    def get_alamat_master_provider(self):
        return self.alamat_master_provider

    def set_status_item_provider(self, status):
        try:
            if self.status != "Direct":
                self.status = status
        except:
            self.status = status


    def get_status_item_provider(self):
        return self.status

    def set_processed(self, found):
        self.found = found

    def is_processed(self):
        return self.found

    def set_alamat_ratio(self, alamat_ratio):
        self.alamat_ratio = alamat_ratio

    def get_alamat_ratio(self):
        return self.alamat_ratio

    def set_ratio(self, ratio):
        self.ratio = ratio

    def get_ratio(self):
        return self.ratio

    def set_total_score(self, total_score):
        self.total_score = total_score

    def get_total_score(self):
        total_score = 0
        try:
            total_score = float(self.total_score)
        except Exception as e:
            print(str(e))
        return float(total_score)

    def set_id(self, pk):
        self.id = pk

    def get_id(self):
        return self.id

    def set_id_model(self, id_model):
        self.id_model = id_model

    def get_id_model(self):
        return self.id_model

    def get_ri(self):
        return self.ri

    def get_rj(self):
        return self.rj

    def set_alamat_prediction(self, alamat_prediction):
        self.alamat_prediction = str(alamat_prediction)

    def get_alamat_prediction(self):
        return self.alamat_prediction

    def set_mapped_times(self, times):
        self.mapped_times = times

    def get_mapped_times(self):
        return self.mapped_times

    def set_count_label_name(self, count):
        self.count_label_name = count

    def set_selected(self, selected):
        self.selected = selected

    def set_nama_asuransi(self, nama_asuransi):
        self.nama_asuransi = nama_asuransi

    def get_count_label_name(self):
        return self.count_label_name

    def get_nama_asuransi(self):
        return self.nama_asuransi

    def get_nama_provider(self):
        return self.nama_provider

    def get_alamat(self):
        return self.alamat

    def get_label_name(self):
        return self.label_name.strip()

    def get_proba_score(self):
        return float(self.proba_score)

    def get_selected(self):
        return self.selected

    def save_item_provider(self):
        print("save item providewr")
        # # # Save Perbandingan Model

        self.save()

    def set_provider_name(self, value):
        self.nama_provider = value

    def set_alamat(self, param):
        self.alamat = param

    def set_nama_alamat(self):
        self.nama_alamat = self.get_nama_provider() + "#" + self.get_alamat()

    def get_nama_alamat(self):
        return self.nama_alamat

    def set_label_name(self, y_preds):
        self.label_name = y_preds

    def set_proba_score(self, nil):
        self.proba_score = nil

    def set_ri(self, param):
        if param.lower() == "y":
            self.ri = 1
        if param.lower() == "n":
            self.ri = 0

    def set_rj(self, param):
        if param.lower() == "y":
            self.rj = 1
        if param.lower() == "n":
            self.rj = 0

    def set_id_asuransi(self, param):
        self.id_asuransi = param

    def get_id_asuransi(self):
        return self.id_asuransi

    def set_validity(self):
        if self.get_status_item_provider() == "Master" or self.get_status_item_provider() == "Ratio" :
            if self.get_total_score() >=60:
                self.validity = True
            else:
                self.validity = False

        else:
            if self.get_total_score() >= 70:
                self.validity = True
            else:
                self.validity = False

    def is_valid(self):
        return self.validity

    def set_saved_in_golden_record(self, bool):
        self.saved_in_golden_record = bool

    def get_saved_in_golden_record(self):
        return self.saved_in_golden_record


class FinalResult(models.Model):
    file_final_location_result = models.CharField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_file_final_location_result(self, nama_file):
        self.file_final_location_result = "media/" + nama_file

    def set_final_result_dataframe(self, dataframe):
        self.final_result_dataframe = dataframe

    def get_final_result_dataframe(self):
        return self.final_result_dataframe


class MasterMatchProcess(models.Model):
    id_file_result = models.CharField(max_length=500)
    id_master_match_file_result = models.CharField(max_length=500)
    match_percentage = models.DecimalField(max_digits=5, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_master_data(self, master_data):
        self.master_data = master_data

    def get_master_data(self):
        return self.master_data

    def create_file_final_result_master_match(self):
        self.file_final_result_master = FinalResult()

    def get_file_final_result_master_match(self):
        return self.file_final_result_master

    def set_file_result_match_processed(self, file_result):
        self.file_result_match = file_result

    def get_file_result_match_processed(self):
        return self.file_result_match

    def insert_into_end_point_andika_assistant_item_provider(self):
        df_final = self.file_final_result_master.get_final_result_dataframe()
        dataframe_insert_new = df_final.loc[df_final['Validity'] == True]
        self.file_result = self.get_file_result_match_processed()
        provider = self.file_result.get_processed_provider()
        id_asuransi = provider.get_id_asuransi()
        print(id_asuransi)
        # url = 'https://www.asateknologi.id/api/inshos'
        # for index, row in dataframe_insert_new.iterrows():
        #     myobj = {'hospitalId': row['IdMaster'], 'insuranceId': id_asuransi, 'outpatient': row['RJ'],
        #              'inpatient': row['RI']}
        #     try:
        #         x = requests.post(url, json=myobj)
        #     except Exception as e:
        #         print(str(e))
        #
        # pass

    def delete_provider_item_hospital_insurances_with_id_insurances(self):
        self.file_result = self.get_file_result_match_processed()
        provider = self.file_result.get_processed_provider()
        id_asuransi = provider.get_id_asuransi()
        url = 'https://www.asateknologi.id/api/inshos-del'
        myobj = {'id_insurance': id_asuransi}
        try:
            x = requests.post(url, json=myobj)
        except Exception as e:
            print(str(e))

    def process_master_matching(self):
        self.create_file_final_result_master_match()
        self.file_result = self.get_file_result_match_processed()
        provider = self.file_result.get_processed_provider()

        # # # # # combine the result with id
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
        list_item_ratio = []
        list_item_total_score = []
        list_item_alamat_ratio = []
        list_item_validity = []
        list_item_status = []

        for item in provider.get_list_item_provider():
            item.set_processed(False)
            # item.set_validity()

            for item_master in self.master_data.get_list_item_master_provider():
                if item.get_label_name() == item_master.get_nama_master():
                    if item.get_total_score() >= 70:
                        item.set_processed(True)
                        id_master_list.append(item_master.get_id_master())
                        provider_name_master_list.append(item_master.get_nama_master())
                        alamat_master_list.append(item_master.get_alamat_master())

                        list_item_provider_nama.append(item.get_nama_provider())
                        list_item_provider_alamat.append(item.get_alamat())
                        list_item_provider_ri.append(item.get_ri())
                        list_item_provider_rj.append(item.get_rj())
                        list_item_provider_score.append(item.get_proba_score())
                        list_item_ratio.append(item.get_ratio())
                        list_item_alamat_ratio.append(item.get_alamat_ratio())

                        list_item_total_score.append(item.get_total_score())

                        item.set_status_item_provider("Master")
                        item.set_validity()

                        list_item_status.append(item.get_status_item_provider())
                        list_item_validity.append(item.is_valid())

                    break

            if item.is_processed() is False:
                item.set_total_score(0)
                item.set_ratio(0)
                item.set_alamat_ratio(0)
            for item_master in self.master_data.get_list_item_master_provider():
                if item.is_processed() is False:
                    ratio_nama = fuzz.ratio(item.get_label_name(), item_master.get_nama_master().strip())
                    ratio_alamat = fuzz.ratio(item.get_alamat(), item_master.get_alamat_master().strip())
                    if item.get_proba_score() != 0 :
                        nilai = ((item.get_proba_score() * 100) + ratio_nama + ratio_alamat) / 3
                    else:
                        nilai = (ratio_nama+ratio_alamat)/2
                    total_ratio_extension = float("{:.2f}".format(nilai))
                    item.set_status_item_provider("Ratio")

                    if float(item.get_total_score()) < total_ratio_extension or item.get_total_score == 0:
                        item.set_ratio(ratio_nama)
                        item.set_alamat_ratio(ratio_alamat)
                        item.set_total_score(total_ratio_extension)
                        item.set_id_master(item_master.get_id_master())
                        item.set_nama_master_provider(item_master.get_nama_master())
                        item.set_alamat_master_provider(item_master.get_alamat_master())

            if item.get_total_score() >= 55 and item.is_processed() is False:

                id_master_list.append(item.get_id_master())
                provider_name_master_list.append(item.get_nama_master_provider())
                alamat_master_list.append(item.get_alamat_master_provider())
                list_item_provider_nama.append(item.get_nama_provider())
                list_item_provider_alamat.append(item.get_alamat())
                list_item_provider_ri.append(item.get_ri())
                list_item_provider_rj.append(item.get_rj())
                list_item_provider_score.append(item.get_proba_score())
                list_item_ratio.append(item.get_ratio())
                list_item_alamat_ratio.append(item.get_alamat_ratio())

                list_item_total_score.append(item.get_total_score())
                item.set_validity()
                list_item_status.append(item.get_status_item_provider())

                list_item_validity.append(item.is_valid())

            elif item.get_total_score() < 55 and item.is_processed() is False:
                id_master_list.append(item.get_id_master())
                provider_name_master_list.append(item.get_nama_master_provider())
                alamat_master_list.append(item.get_alamat_master_provider())
                list_item_provider_nama.append(item.get_nama_provider())
                list_item_provider_alamat.append(item.get_alamat())
                list_item_provider_ri.append(item.get_ri())
                list_item_provider_rj.append(item.get_rj())
                list_item_provider_score.append(item.get_proba_score())
                list_item_ratio.append(ratio_nama)
                list_item_alamat_ratio.append(ratio_alamat)

                list_item_total_score.append(item.get_total_score())
                item.set_validity()
                list_item_status.append(item.get_status_item_provider())

                list_item_validity.append(item.is_valid())

        dict_result = {
            'IdMaster': pd.Series(id_master_list),
            'Master_Nama': pd.Series(provider_name_master_list),
            'Master_Alamat': pd.Series(alamat_master_list),
            'Nama': pd.Series(list_item_provider_nama),
            'Alamat': pd.Series(list_item_provider_alamat),
            'Score': pd.Series(list_item_provider_score),
            'Ratio': pd.Series(list_item_ratio),
            'Alamat_Ratio': pd.Series(list_item_alamat_ratio),
            'Total_Score': pd.Series(list_item_total_score),
            'RI': pd.Series(list_item_provider_ri),
            'RJ': pd.Series(list_item_provider_rj),
            'Validity': pd.Series(list_item_validity),
            'Status': pd.Series(list_item_status)
        }
        self.processed_dataframe = pd.DataFrame(dict_result)

        nama_file = provider.get_nama_asuransi_model() + "_result_final.xlsx"

        writer = pd.ExcelWriter('media/' + nama_file, engine='xlsxwriter')
        # # # Convert the dataframe to an XlsxWriter Excel object.
        self.processed_dataframe.to_excel(writer, sheet_name='Sheet1', index=False)

        writer.close()

        self.file_final_result_master.set_file_final_location_result(nama_file)
        self.file_final_result_master.save()
        self.file_final_result_master.set_final_result_dataframe(self.processed_dataframe)

    def get_final_result_dataframe(self):
        return self.processed_dataframe

    def save_matching_information(self):
        self.id_file_result = self.file_result_match.pk
        self.id_master_match_file_result = self.file_final_result_master.pk
        self.match_percentage = 1.00
        self.save()


class FileResult(models.Model):
    file_location_result = models.CharField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_processed_provider(self, provider):
        self.provider = provider
        self.set_nama(provider.get_nama_asuransi_model())

    def get_processed_provider(self):
        return self.provider

    def set_file_location_result(self, nama_file):
        self.file_location_result = "media/" + nama_file

    def save_file_result_information(self):
        self.match_percentage = 0.00
        self.save()

    def get_file_location_result(self):
        return self.file_location_result

    def set_nama(self, param):
        self.nama_file = param

    def get_nama(self):
        return self.nama_file


class MatchProcess(models.Model):
    id_model = models.CharField(max_length=500)
    id_file_result = models.CharField(max_length=500)
    match_percentage = models.DecimalField(max_digits=5, decimal_places=2)
    status_finish = models.CharField(max_length=8)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_golden_record_instance(self, golden_record):
        self.golden_record = golden_record

    def get_golden_record_instance(self):
        return self.golden_record

    def set_file_result(self, file_result):
        self.file_result = file_result

    def get_file_result(self):
        return self.file_result

    def set_status_finish(self, status_finish):
        self.status_finish = status_finish

    def set_master_data(self, master_data):
        self.master_data = master_data

    def get_master_data(self):
        return self.master_data

    def get_status_finish(self):
        return self.status_finish

    def get_provider_list_json(self):
        ls = []
        for provider in self.get_provider_list():
            del provider._state
            ls.append(provider.__dict__)
        return ls

    def get_a_provider_from_id(self, pk):
        id_provider = pk
        for data in self.get_provider_list():
            if data.get_primary_key_provider() == str(id_provider):
                return data

    def set_list_provider(self):
        self.list_provider = List_Processed_Provider()
        # self.list_provider.set_empty_provider_list()

    def get_list_provider(self):
        return self.list_provider

    def set_dataset(self):
        self.dataset = Dataset(pd)

    def get_dataset(self):
        return self.dataset

    def set_dfhandler(self):
        self.df_handler = DFHandler()

    def get_dfhandler(self):
        return self.df_handler

    def map_list_item_provider_to_column_output(self, provider_item_list):
        print("Map List Item To Column Output")
        self.column_output = ColumnOutput()

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

    def error_value_write(self, item_provider, e):
        item_provider.set_label_name("-")
        item_provider.set_proba_score(0)
        item_provider.set_count_label_name(0)
        item_provider.set_alamat_prediction("-")

        # item_provider.save()
        print("Error read the file " + item_provider.get_nama_provider() + " " + str(e))

    def vectorize_text(self, text, tfidf_vec):
        # text = "Klinik Ananda"
        my_vec = tfidf_vec.transform([text])
        return my_vec.toarray()

    def start(self):
        self.provider_name_predict_list = []
        filename = 'tfidf_vec.pickle'
        self.tfidf_vec1 = pickle.load(open(filename, 'rb'))
        filename = 'finalized_model.sav'
        self.loaded_model1 = pickle.load(open(filename, 'rb'))
        self.ex = ExcelBacaTulis()
        self.provider_list = []
        self.df = None
        self.column = ColumnToRead()

    def process_matching(self):
        # get lokasi excel
        print("Match Process")

        self.set_dataset()
        self.file_result = FileResult()

        self.processed_provider = self.list_provider.get_provider_list().pop()
        self.processed_provider.save_perbandingan_model()

        # Compare and save the Item !
        pd.options.display.max_colwidth = None
        df_dataset_non_duplicate = self.get_dataset().get_dataframe_after_cleaned_no_duplicate()
        self.golden_record.set_golden_record_list_item()
        golden_record_list_item = self.golden_record.get_golden_record_list_item()
        print("Compare to golden record list item")

        for item_provider in self.processed_provider.get_list_item_provider():
            try:
                for item_golden in golden_record_list_item:

                    if item_golden.get_nama_provider() == item_provider.get_nama_provider():
                        item_provider.set_id_model(self.processed_provider.get_primary_key_provider())
                        item_provider.set_label_name(item_golden.get_label_name())
                        item_provider.set_proba_score(item_golden.get_proba_score())
                        item_provider.set_ratio(item_golden.get_ratio())

                        item_provider.set_total_score(item_golden.get_total_score())

                        item_provider.set_alamat_ratio(item_golden.get_alamat_ratio())
                        item_provider.set_count_label_name(0)
                        item_provider.set_alamat_prediction(item_golden.get_alamat_prediction())
                        item_provider.set_golden_record(1)
                        item_provider.set_saved_in_golden_record(True)
                        item_provider.set_compared(True)
                        break

                if item_provider.get_golden_record_status() == "" and item_provider.get_saved_in_golden_record() is not True:
                    item_provider.set_golden_record(0)

                    nama_alamat = item_provider.get_nama_alamat()
                    sample1 = self.vectorize_text(nama_alamat, self.tfidf_vec1)
                    y_preds = self.loaded_model1.predict(sample1)

                    # add prediction ke list
                    y_preds = str(y_preds).replace("[", "").replace("]", "").replace("'", "")

                    # calculate proba
                    p = self.loaded_model1.predict_proba(sample1)
                    ix = p.argmax(1).item()
                    nil = float("{:.2f}".format(p[0, ix]))
                    score = fuzz.ratio(item_provider.get_nama_provider(), y_preds)

                    # jika prediksi sama dengan dataset maka ambil alamat dari dataset
                    val_master = (df_dataset_non_duplicate['subject'].eq(y_preds))

                    res_master = df_dataset_non_duplicate[val_master]
                    al = res_master["alamat"].head(1)

                    item_provider.set_id_model(self.processed_provider.get_primary_key_provider())
                    item_provider.set_label_name(y_preds)
                    item_provider.set_proba_score(nil)
                    item_provider.set_ratio(score)

                    alamat_ratio = fuzz.token_set_ratio(al.values[0], item_provider.get_alamat())
                    total_score = float("{:.2f}".format((score + (nil * 100) + alamat_ratio) / 3))
                    item_provider.set_total_score(total_score)
                    if total_score < 70:
                        item_provider.set_compared(False)
                    else:
                        item_provider.set_compared(True)

                    item_provider.set_alamat_ratio(alamat_ratio)
                    item_provider.set_count_label_name(0)
                    item_provider.set_alamat_prediction(al.values[0])
                    item_provider.set_golden_record(0)
                    if total_score >= 70:
                        item_provider.save()
                    item_provider.set_saved_in_golden_record(False)

                for item_master in self.master_data.get_list_item_master_provider():
                    if item_provider.get_compared() is not True:
                        score = fuzz.ratio(item_provider.get_nama_provider(), item_master.get_nama_master())
                        alamat_ratio = fuzz.token_set_ratio(item_master.get_alamat_master(), item_provider.get_alamat())
                        total_score = float("{:.2f}".format((score + alamat_ratio) / 2))

                        if item_provider.get_total_score() <= total_score:
                            # print(item_provider.get_nama_provider(),item_master.get_nama_master(),score,alamat_ratio)
                            item_provider.set_total_score(total_score)
                            item_provider.set_id_model(self.processed_provider.get_primary_key_provider())
                            item_provider.set_label_name(item_master.get_nama_master())
                            item_provider.set_proba_score(0)
                            item_provider.set_ratio(score)
                            item_provider.set_alamat_ratio(alamat_ratio)
                            item_provider.set_count_label_name(0)
                            item_provider.set_alamat_prediction(item_master.get_alamat_master())
                            item_provider.set_golden_record(0)
                            item_provider.set_saved_in_golden_record(False)
                            item_provider.set_status_item_provider("Direct")
                            item_provider.save()
                item_provider.set_compared(True)



            except Exception as e:
                # self.error_value_write(item_provider, e)
                print(str(e))

        # # # map processed dataframe column to output desired column
        mapped = self.map_list_item_provider_to_column_output(self.processed_provider.get_list_item_provider())

        # # # convert mapped list to dataframe
        self.result_dataframe = pd.DataFrame(mapped)

        # set processed provider
        self.file_result.set_processed_provider(self.processed_provider)

    def create_file_result(self):

        # # get nama asuransi
        nama_asuransi = self.file_result.get_nama()
        nama_file = nama_asuransi + "_result.xlsx"
        # # # Declare write
        writer = pd.ExcelWriter('media/' + nama_file,
                                engine='xlsxwriter')
        # # # Convert the dataframe to an XlsxWriter Excel object.
        self.result_dataframe.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.close()

        self.file_result.set_file_location_result(nama_file)
        self.file_result.save_file_result_information()
        self.save_matching_information()

    def save_matching_information(self):
        self.id_file_result = self.file_result.pk
        self.id_model = self.processed_provider.get_primary_key_provider()
        self.match_percentage = 1.00
        self.save()

    def set_id(self, pk):
        self.id = pk

    def get_id(self):
        return self.id

    def set_id_model(self, id_model):
        self.id_model = id_model

    def get_id_model(self):
        return self.id_model

    def set_id_file_result(self, id_file_result):
        self.id_file_result = id_file_result

    def get_id_file_result(self):
        return self.id_file_result

    def set_created_at(self, created_at):
        self.created_at = created_at

    def get_created_at(self):
        return self.created_at


class GoldenRecordMatch(models.Model):
    id_model = models.CharField(max_length=500)
    id_finalresult = models.CharField(max_length=500)
    total_golden_record = models.CharField(max_length=500)
    nama_provider = models.CharField(max_length=500)
    alamat = models.CharField(max_length=500)
    total_ratio = models.CharField(max_length=500)
    status = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_golden_record_list_item(self):

        self.golden_record_list_item = ItemProvider.objects.raw(
            "select * from model_itemprovider where golden_record = 1")

    def get_golden_record_list_item(self):
        print("golden record list item")
        return self.golden_record_list_item

    def set_master_match_process(self):
        self.master_match_process = MasterMatchProcess()

    def get_master_match_process(self):
        return self.master_match_process

    def set_final_result(self, finalresult):
        self.final_result = finalresult

    def set_file_result(self, fileresult):
        self.file_result = fileresult
        self.provider = self.file_result.get_processed_provider()

    def get_final_result(self):
        return self.final_result

    def get_file_result(self):
        return self.file_result

    def set_processed_provider(self):
        self.provider = self.file_result.get_processed_provider()

    def process_golden_record(self):
        self.list_item_provider = self.provider.get_list_item_provider()
        df_final = self.final_result.get_final_result_dataframe()
        df1 = df_final.loc[df_final['Validity'] == True]
        df_convert_to_int = df1.astype({'Total_Score': 'float'})
        df2 = df_convert_to_int.loc[df_convert_to_int['Status'].eq("Master") | (
                    df_convert_to_int['Total_Score'].ge(90) & df_convert_to_int['Status'].eq("Ratio"))]
        # df2 = df_convert_to_int.loc[df_convert_to_int['Status'].eq("Master")]
        # df2 = df_convert_to_int[df_convert_to_int['Total_Score'] >= 90 | df_convert_to_int['Status'] == "Master"]
        for row in df2.itertuples(index=True, name='Sheet1'):
            for item_provider in self.list_item_provider:

                if row.Nama == item_provider.get_nama_provider():
                    if item_provider.get_saved_in_golden_record() is not True:
                        item_provider.set_golden_record(1)

                        item_provider.save()
                        golden = GoldenRecordMatch()
                        golden.id_model = self.provider.pk
                        golden.id_finalresult = self.final_result.pk
                        golden.nama_provider = item_provider.get_nama_provider()
                        golden.alamat = item_provider.get_alamat()
                        golden.status = item_provider.get_status_item_provider()
                        golden.total_ratio = item_provider.get_total_score()
                        golden.save()

        pass


class Provider(models.Model):
    nama_asuransi = models.CharField(max_length=500)
    file_location = models.CharField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)

    @staticmethod
    def get_model_from_filter(nama_asuransi):
        mydata = Provider.objects.filter(nama_asuransi__contains=nama_asuransi).order_by('created_at').first()
        if not mydata:
            return False
        return mydata

    def set_status_matching(self, status):
        self.status_matching = status

    def get_status_matching(self):
        return self.status_matching

    def set_created_at(self, created_at):
        self.created_at = created_at

    def get_created_at(self):
        return self.created_at

    def get_dataframe(self):
        df = pd.read_excel(self.file_location)
        # clean the dataframe
        pembersih = Pembersih(df)

        df = pembersih._return_df()
        return df

    def get_nama_asuransi_model(self):
        return self.nama_asuransi

    def set_nama_asuransi_model(self, nama_asuransi):
        self.nama_asuransi = nama_asuransi

    def set_id_asuransi_model(self, id_asuransi):
        self.id_asuransi = id_asuransi

    def get_id_asuransi(self):
        return self.id_asuransi

    def set_id(self, pk):
        self.id = pk

    def get_file_location_result(self):
        return "media" + self.file_location_result

    def get_lokasi_excel_pembanding(self):
        print("Get Excel Pembanding")
        return self.file_location

    def set_file_location(self, file_location):
        self.file_location = file_location

    def link_to_item_list(self):
        provider_item_list = []
        for row in self.get_dataframe().itertuples(index=True, name='Sheet1'):
            provider_object = ItemProvider()
            provider_object.set_id_asuransi(self.get_id_asuransi())
            provider_object.set_id_model(self.pk)
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
            provider_object.set_saved_in_golden_record(False)

            provider_item_list.append(provider_object)

        self.set_list_item_provider(provider_item_list)

    def get_primary_key_provider(self):
        return str(self.pk)

    def save_perbandingan_model(self):
        # # # Save Perbandingan Model
        nama_asuransi = self.get_nama_asuransi_model()
        self.nama_asuransi = nama_asuransi
        self.file_location_result = "/" + nama_asuransi + "_result.xlsx"
        self.save()

    def set_list_item_provider(self, list_item_provider):
        self.list_item_provider = list_item_provider

    def get_list_item_provider(self):
        return self.list_item_provider

    def get_list_item_provider_json(self):
        return self.list_item_provider_json

    def set_list_item_provider_json(self, list_item_provider):
        self.list_item_provider_json = list_item_provider
