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
    stateId = models.CharField(max_length=2)
    cityId = models.CharField(max_length=2)
    category_1 = models.CharField(max_length=2)
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
        if self.golden_record != 1:
            return 0
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
        try:
            return self.status
        except Exception as e:
            return None

    def set_processed(self, found):
        self.found = found

    def is_processed(self):
        return self.found

    def set_alamat_ratio(self, alamat_ratio):
        self.alamat_ratio = alamat_ratio

    def get_alamat_ratio(self):
        try:
            self.alamat_ratio = float(self.alamat_ratio)
        except:
            self.alamat_ratio = 0
        return self.alamat_ratio

    def set_ratio(self, ratio):
        try:
            self.ratio = float(ratio)
        except:
            self.ratio = 0

    def get_ratio(self):
        try:
            self.ratio = float(self.ratio)
        except:
            self.ratio = 0
        return self.ratio

    def set_total_score(self, total_score):
        try:
            self.total_score = float(total_score)
        except:
            self.total_score = 0

        if self.total_score < 70 and self.golden_record == 0:
            self.set_compared(False)
        else:
            self.set_compared(True)

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

        return self.nama_provider.strip().lower()

    def get_alamat(self):
        return self.alamat

    def get_label_name(self):
        return self.label_name.strip().lower()

    def get_proba_score(self):
        return float(self.proba_score)

    def get_selected(self):
        return self.selected

    def save_item_provider(self):
        print("save item providewr")
        # # # Save Perbandingan Model

        self.save()

    def set_provider_name(self, value):
        remove_words = ["rsia", "rsu", "rumah sakit", "rs", "optik", "klinik", "clinic", "laboratorium", "lab", "optic",
                        ".", ","]
        # remove_words = []
        for rem in remove_words:
            value = value.replace(rem, " ")
        self.nama_provider = value.strip()

    def set_alamat(self, param):
        self.alamat = param

    def set_nama_alamat(self):
        self.nama_alamat = self.get_nama_provider() + " " + self.get_alamat()

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
        if self.get_status_item_provider() == "Master":
            # if self.get_alamat_ratio() >= 85 or self.get_ratio() >= 95 or self.get_proba_score() > 0.8:
            self.validity = True


        elif self.get_status_item_provider() == "PROBA":
            if self.get_ratio() > 80:
                self.validity = True
            else:
                self.validity = False

        elif self.get_status_item_provider() == "Ratio":
            if self.get_ratio() >= 90 or self.get_alamat_ratio() > 95:
                self.validity = True
            else:
                self.validity = False
        elif self.get_status_item_provider() == "Direct 1":
            if self.get_ratio() >= 90 :
                self.validity = True
            else:
                self.validity = False

        elif self.get_status_item_provider() == "Direct 2":
            if  (self.get_alamat_ratio() >= 80 and self.get_proba_score() > 0.40) or self.get_alamat_ratio() > 95:
                self.validity = True
            else:
                self.validity = False

        else:

            self.validity = False

    def is_validity(self):
        return self.validity

    def set_saved_in_golden_record(self, bool):
        self.saved_in_golden_record = bool

    def get_saved_in_golden_record(self):
        if self.saved_in_golden_record is not True:
            return False
        return self.saved_in_golden_record

    def set_city_id(self, city_id):
        self.cityId = city_id

    def get_city_id(self):
        return self.cityId

    def set_state_id(self, state_id):
        self.stateId = state_id

    def get_state_id(self):
        return self.stateId


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
        print("FINAL RESULT MASTER")
        print(self.file_final_result_master.file_final_location_result)
        return self.file_final_result_master

    def set_file_result_match_processed(self, file_result):
        self.file_result_match = file_result

    def get_file_result_match_processed(self):
        return self.file_result_match

    def insert_into_end_point_andika_assistant_item_provider(self):
        df_final = self.file_final_result_master.get_final_result_dataframe()
        dataframe_insert_new = df_final.loc[df_final['Validity'] == True]

        dataframe_insert_master = df_final.loc[df_final['Validity'] == False]

        df_convert_to_int = dataframe_insert_master.astype({'Total_Score': 'float'})
        df2 = df_convert_to_int.loc[df_convert_to_int['Total_Score'].lt(60)]

        self.file_result = self.get_file_result_match_processed()
        provider = self.file_result.get_processed_provider()

        # for item_provider in provider.get_list_item_provider():
        #     for index, row in df2.iterrows():
        #         if row['Alamat'] == item_provider.get_alamat():
        #             print("Found")
        #             print(item_provider.get_nama_provider(),item_provider.get_state_id())
        #             url = 'https://www.asateknologi.id/api/master'
        #             myobj = {'stateId': item_provider.get_state_id(),
        #                      'cityId': item_provider.get_city_id(),
        #                      'category1': 2,
        #                      'provider_name': item_provider.get_nama_provider(),
        #                      'address': item_provider.get_alamat(),
        #                      'tel':'021',
        #                      'inpatient':0,
        #                      'outpatient':0}
        #             try:
        #                 x = requests.post(url, json=myobj)
        #             except Exception as e:
        #                 print(str(e))
        #
        #             pass

        id_asuransi = provider.get_id_asuransi()
        print(id_asuransi)
        url = 'https://www.asateknologi.id/api/inshos'
        for index, row in dataframe_insert_new.iterrows():
            myobj = {'hospitalId': row['IdMaster'], 'insuranceId': id_asuransi, 'outpatient': row['RJ'],
                     'inpatient': row['RI']}
            try:
                x = requests.post(url, json=myobj)
            except Exception as e:
                print(str(e))

        pass

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

    def add_item_has_status_master_to_list(self,item):
        if item.get_status_item_provider() == "Master":
            item.set_processed(True)
            self.id_master_list.append(item.get_id_master())
            self.provider_name_master_list.append(item.get_nama_master_provider())
            self.alamat_master_list.append(item.get_alamat_master_provider())

            self.list_item_provider_nama.append(item.get_nama_provider())
            self.list_item_provider_alamat.append(item.get_alamat())
            self.list_item_provider_ri.append(item.get_ri())
            self.list_item_provider_rj.append(item.get_rj())
            self.list_item_provider_score.append(item.get_proba_score())
            self.list_item_ratio.append(item.get_ratio())
            self.list_item_alamat_ratio.append(item.get_alamat_ratio())

            self.list_item_total_score.append(item.get_total_score())

            item.set_status_item_provider("Master")

            self.list_item_status.append(item.get_status_item_provider())
            item.set_validity()
            self.list_item_validity.append(item.is_validity())



    def compare_golden_record_to_master(self, item):
        # # Comparing golden record
        for item_master in self.master_data.get_list_item_master_provider():
            if item.get_saved_in_golden_record() is True:
                if item.get_label_name() == item_master.get_nama_master():
                    item.set_processed(True)
                    self.id_master_list.append(item_master.get_id_master())
                    self.provider_name_master_list.append(item_master.get_nama_master())
                    self.alamat_master_list.append(item_master.get_alamat_master())

                    self.list_item_provider_nama.append(item.get_nama_provider())
                    self.list_item_provider_alamat.append(item.get_alamat())
                    self.list_item_provider_ri.append(item.get_ri())
                    self.list_item_provider_rj.append(item.get_rj())
                    self.list_item_provider_score.append(item.get_proba_score())
                    self.list_item_ratio.append(item.get_ratio())
                    self.list_item_alamat_ratio.append(item.get_alamat_ratio())

                    self.list_item_total_score.append(item.get_total_score())

                    item.set_status_item_provider("Master")

                    self.list_item_status.append(item.get_status_item_provider())
                    item.set_validity()
                    self.list_item_validity.append(item.is_validity())

                    break

        # # define is it true or false to proceed to next step
        item.set_validity()

    def if_proba_score_high_add_to_list(self, item, list_item_master_provider):
        # if validity still false
        if item.get_status_item_provider() != "Master" and (item.get_proba_score() > 0.3 or item.get_alamat_ratio() >= 95):
            item.set_total_score(0)
            item.set_ratio(0)

            for item_master in list_item_master_provider:
                if item.get_label_name() == item_master.get_nama_master():
                    item.set_status_item_provider("PROBA")
                    item.set_ratio(100)
                    item.set_total_score(100)
                    item.set_id_master(item_master.get_id_master())
                    item.set_nama_master_provider(item_master.get_nama_master())
                    item.set_alamat_master_provider(item_master.get_alamat_master())
                    break

            # # add to list if they are true
            item.set_validity()
            if item.is_validity():
                self.id_master_list.append(item.get_id_master())
                self.provider_name_master_list.append(item.get_nama_master_provider())
                self.alamat_master_list.append(item.get_alamat_master_provider())
                self.list_item_provider_nama.append(item.get_nama_provider())
                self.list_item_provider_alamat.append(item.get_alamat())
                self.list_item_provider_ri.append(item.get_ri())
                self.list_item_provider_rj.append(item.get_rj())
                self.list_item_provider_score.append(item.get_proba_score())
                self.list_item_ratio.append(item.get_ratio())
                self.list_item_alamat_ratio.append(item.get_alamat_ratio())

                self.list_item_total_score.append(item.get_total_score())
                self.list_item_status.append(item.get_status_item_provider())

                self.list_item_validity.append(item.is_validity())

    def if_proba_score_lower_than_threshold(self, item, list_item_master_provider):
        # IF PROBA SCORE BELOW THRESHOLD , THEN DO RATIO CALCULATION
        # if validity still false
        # if item.is_validity() is False and (item.get_proba_score() > 0.3 and item.get_proba_score() <= 0.8):
        if item.is_validity() is False :

            item.set_total_score(0)
            item.set_ratio(0)

            for item_master in list_item_master_provider:
                ratio_nama = fuzz.ratio(item.get_nama_provider(), item_master.get_nama_master())
                total_ratio_extension = float("{:.2f}".format(ratio_nama))

                if float(item.get_total_score()) < total_ratio_extension:
                    item.set_label_name(item_master.get_nama_master())
                    item.set_ratio(ratio_nama)
                    item.set_total_score(total_ratio_extension)
                    item.set_id_master(item_master.get_id_master())
                    item.set_nama_master_provider(item_master.get_nama_master())
                    item.set_alamat_master_provider(item_master.get_alamat_master())

            # # add to list if they are true
            item.set_status_item_provider("Ratio")
            item.set_validity()
            if item.is_validity():
                self.id_master_list.append(item.get_id_master())
                self.provider_name_master_list.append(item.get_nama_master_provider())
                self.alamat_master_list.append(item.get_alamat_master_provider())
                self.list_item_provider_nama.append(item.get_nama_provider())
                self.list_item_provider_alamat.append(item.get_alamat())
                self.list_item_provider_ri.append(item.get_ri())
                self.list_item_provider_rj.append(item.get_rj())
                self.list_item_provider_score.append(item.get_proba_score())
                self.list_item_ratio.append(item.get_ratio())
                self.list_item_alamat_ratio.append(item.get_alamat_ratio())

                self.list_item_total_score.append(item.get_total_score())
                self.list_item_status.append(item.get_status_item_provider())

                self.list_item_validity.append(item.is_validity())

    def compare_nama_provider_to_master(self, item):
        if item.is_validity() is False:
            item.set_total_score(0)
            item.set_ratio(0)

            for item_master in self.master_data.get_list_item_master_provider():
                score = fuzz.ratio(item.get_nama_provider(), item_master.get_nama_master())
                # alamat_ratio = fuzz.ratio(item.get_alamat_master(), item.get_alamat())
                total_score = float("{:.2f}".format((score)))

                if item.get_total_score() <= total_score:
                    item.set_total_score(total_score)
                    item.set_label_name(item_master.get_nama_master())
                    item.set_ratio(score)
                    print(item.get_label_name(), item_master.get_nama_master(), ratio_nama)

                    # item.set_alamat_ratio(alamat_ratio)
                    item.set_alamat_prediction(item_master.get_alamat_master())
                    item.set_status_item_provider("Direct 1")
                    item.set_id_master(item_master.get_id_master())
                    item.set_nama_master_provider(item_master.get_nama_master())
                    item.set_alamat_master_provider(item_master.get_alamat_master())

            # set validity
            item.set_validity()
            if (item.is_validity()):
                self.id_master_list.append(item.get_id_master())
                self.provider_name_master_list.append(item.get_nama_master_provider())
                self.alamat_master_list.append(item.get_alamat_master_provider())
                self.list_item_provider_nama.append(item.get_nama_provider())
                self.list_item_provider_alamat.append(item.get_alamat())
                self.list_item_provider_ri.append(item.get_ri())
                self.list_item_provider_rj.append(item.get_rj())
                self.list_item_provider_score.append(item.get_proba_score())
                self.list_item_ratio.append(item.get_ratio())
                self.list_item_alamat_ratio.append(item.get_alamat_ratio())

                self.list_item_total_score.append(item.get_total_score())
                self.list_item_status.append(item.get_status_item_provider())

                self.list_item_validity.append(item.is_validity())


    def compare_alamat_provider_to_master(self, item,list_item_master_provider):
        if item.is_validity() is False:
            item.set_total_score(0)
            item.set_alamat_ratio(0)

            for item_master in list_item_master_provider:
                alamat_ratio = fuzz.ratio(item.get_alamat(), item_master.get_alamat_master())
                total_score = float("{:.2f}".format((alamat_ratio)))

                if item.get_total_score() <= total_score :
                    item.set_total_score(total_score)
                    item.set_label_name(item_master.get_nama_master())
                    item.set_alamat_ratio(alamat_ratio)
                    item.set_alamat_prediction(item_master.get_alamat_master())
                    item.set_id_master(item_master.get_id_master())
                    item.set_nama_master_provider(item_master.get_nama_master())
                    item.set_alamat_master_provider(item_master.get_alamat_master())


            item.set_status_item_provider("Direct 2")
            # set validity
            item.set_validity()
            # if (item.is_validity()):
            self.id_master_list.append(item.get_id_master())
            self.provider_name_master_list.append(item.get_nama_master_provider())
            self.alamat_master_list.append(item.get_alamat_master_provider())
            self.list_item_provider_nama.append(item.get_nama_provider())
            self.list_item_provider_alamat.append(item.get_alamat())
            self.list_item_provider_ri.append(item.get_ri())
            self.list_item_provider_rj.append(item.get_rj())
            self.list_item_provider_score.append(item.get_proba_score())
            self.list_item_ratio.append(item.get_ratio())
            self.list_item_alamat_ratio.append(item.get_alamat_ratio())

            self.list_item_total_score.append(item.get_total_score())
            self.list_item_status.append(item.get_status_item_provider())

            self.list_item_validity.append(item.is_validity())

    def process_master_matching(self):
        self.create_file_final_result_master_match()
        self.file_result = self.get_file_result_match_processed()
        provider = self.file_result.get_processed_provider()

        # # # # # combine the result with id
        print("Create Result File With ID")
        # read excel master provider and get the dataframe
        self.id_master_list = []
        self.alamat_master_list = []
        self.provider_name_master_list = []
        self.list_item_provider_nama = []
        self.list_item_provider_alamat = []
        self.list_item_provider_score = []
        self.list_item_provider_ri = []
        self.list_item_provider_rj = []
        self.list_item_ratio = []
        self.list_item_total_score = []
        self.list_item_alamat_ratio = []
        self.list_item_validity = []
        self.list_item_status = []

        list_item_master_provider = self.master_data.get_list_item_master_provider()
        for item in provider.get_list_item_provider():
            item.set_processed(False)
            item.set_validity()
            # FILTERING
            # # 1. First, add that exactly match to master added to master list
            # self.compare_golden_record_to_master(item)
            self.add_item_has_status_master_to_list(item)

            # # 2. Second, if proba score is higher than threshold:
            self.if_proba_score_high_add_to_list(item, list_item_master_provider)

            # # 3. If proba score is lower than threshold
            self.if_proba_score_lower_than_threshold(item, list_item_master_provider)

            # # 4. If fail , compare address to master address
            self.compare_alamat_provider_to_master(item,list_item_master_provider)

            # if item.get_status_item_provider() == "Direct 1" or item.get_status_item_provider() == "Direct 2":
            #     item.set_processed(True)
            #     id_master_list.append(item.get_id_master())
            #     provider_name_master_list.append(item.get_nama_master_provider())
            #     alamat_master_list.append(item.get_alamat_master_provider())
            #
            #     list_item_provider_nama.append(item.get_nama_provider())
            #     list_item_provider_alamat.append(item.get_alamat())
            #     list_item_provider_ri.append(item.get_ri())
            #     list_item_provider_rj.append(item.get_rj())
            #     list_item_provider_score.append(item.get_proba_score())
            #     list_item_ratio.append(item.get_ratio())
            #     list_item_alamat_ratio.append(item.get_alamat_ratio())
            #
            #     list_item_total_score.append(item.get_total_score())
            #
            #     item.set_validity()
            #
            #     list_item_status.append(item.get_status_item_provider())
            #     list_item_validity.append(item.is_valid())
            #
            #     break

        dict_result = {
            'IdMaster': pd.Series(self.id_master_list),
            'Master_Nama': pd.Series(self.provider_name_master_list),
            'Master_Alamat': pd.Series(self.alamat_master_list),
            'Nama': pd.Series(self.list_item_provider_nama),
            'Alamat': pd.Series(self.list_item_provider_alamat),
            'Score': pd.Series(self.list_item_provider_score),
            'Ratio': pd.Series(self.list_item_ratio),
            'Alamat_Ratio': pd.Series(self.list_item_alamat_ratio),
            'Total_Score': pd.Series(self.list_item_total_score),
            'RI': pd.Series(self.list_item_provider_ri),
            'RJ': pd.Series(self.list_item_provider_rj),
            'Validity': pd.Series(self.list_item_validity),
            'Status': pd.Series(self.list_item_status)
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

    def get_category_dict(self):
        kategori_dict = {}
        kategori_dict["RS"] = 1
        kategori_dict["KLINIK"] = 2
        kategori_dict["APOTEK"] = 3
        kategori_dict["LAB"] = 4
        kategori_dict["PRAKTEK"] = 5
        kategori_dict["OPTIK"] = 6

        return kategori_dict

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
        print("{} is loaded".format("tfidf_vec1"))

        filename = 'finalized_model.sav'
        self.loaded_model1 = pickle.load(open(filename, 'rb'))
        print("{} is loaded".format("loaded_model1"))

        self.set_dataset()

        self.ex = ExcelBacaTulis()
        self.provider_list = []
        self.df = None
        self.column = ColumnToRead()

    def compare_to_golden_record(self, item_provider):
        print("Compare to golden record list item")

        golden_record_list_item = self.golden_record.get_golden_record_list_item()



        for item_golden in golden_record_list_item:

            if item_golden.get_nama_provider() == item_provider.get_nama_provider():
                item_provider.set_saved_in_golden_record(True)

                item_provider.set_id_model(self.processed_provider.get_primary_key_provider())
                item_provider.set_label_name(item_golden.get_label_name())
                item_provider.set_proba_score(item_golden.get_proba_score())
                item_provider.set_ratio(item_golden.get_ratio())

                item_provider.set_total_score(item_golden.get_total_score())

                item_provider.set_alamat_ratio(item_golden.get_alamat_ratio())
                item_provider.set_count_label_name(0)
                item_provider.set_alamat_prediction(item_golden.get_alamat_prediction())

                break

    def compare_item_provider_name_to_master_name(self, item_provider, item_master_list):
        print("Compare to master data record list item")

        for item_master in item_master_list:

            if item_master.get_nama_master() == item_provider.get_nama_provider():
                item_provider.set_saved_in_golden_record(True)

                item_provider.set_id_model(self.processed_provider.get_primary_key_provider())
                item_provider.set_label_name(item_master.get_nama_master())
                item_provider.set_nama_master_provider(item_master.get_nama_master())
                item_provider.set_proba_score(1)
                item_provider.set_ratio(100)

                item_provider.set_total_score(100)

                item_provider.set_alamat_ratio(100)
                item_provider.set_count_label_name(0)
                item_provider.set_alamat_prediction(item_master.get_alamat_master())
                item_provider.set_alamat_master_provider(item_master.get_alamat_master())
                item_provider.set_status_item_provider("Master")
                item_provider.set_id_master(item_master.get_id_master())

                break

    def calculate_score(self, item_provider):
        print("Calculate Prediction Score")
        nama_alamat = item_provider.get_nama_alamat()
        sample1 = self.vectorize_text(nama_alamat, self.tfidf_vec1)
        y_preds = self.loaded_model1.predict(sample1)

        print(nama_alamat, y_preds)
        # add prediction ke list
        y_preds = str(y_preds).replace("[", "").replace("]", "").replace("'", "").replace("rs","").replace("lab","").replace(",","").replace(".","")

        # calculate proba
        p = self.loaded_model1.predict_proba(sample1)
        ix = p.argmax(1).item()
        nil = float("{:.2f}".format(p[0, ix]))
        score = fuzz.ratio(item_provider.get_nama_provider(), y_preds)

        score_tuple = (y_preds, nil, score)
        print(score,item_provider.get_nama_provider(),y_preds)
        return score_tuple

    def predict_item_provider(self, item_provider):
        print("Compare not golden record")
        df_dataset_non_duplicate = self.get_dataset().get_dataframe_after_cleaned_no_duplicate()

        # if not golden record
        if item_provider.get_saved_in_golden_record() is False:
            # calculate item provider prediction
            y_preds, nil, score = self.calculate_score(item_provider)

            # set item provider value
            item_provider.set_id_model(self.processed_provider.get_primary_key_provider())
            item_provider.set_label_name(y_preds)
            item_provider.set_proba_score(nil)
            item_provider.set_ratio(score)

            # calculate the fuzzy logic to reinforce the score
            # # # jika prediksi sama dengan dataset maka ambil alamat dari dataset
            val_master = (df_dataset_non_duplicate['subject'].eq(y_preds))
            res_master = df_dataset_non_duplicate[val_master]
            al = res_master["alamat"].head(1)

            alamat_ratio = fuzz.ratio(al.values[0], item_provider.get_alamat())
            total_score = float("{:.2f}".format((score + (nil * 100) + alamat_ratio) / 3))

            item_provider.set_total_score(total_score)

            item_provider.set_alamat_ratio(alamat_ratio)
            item_provider.set_count_label_name(0)
            item_provider.set_alamat_prediction(al.values[0])

            item_provider.set_validity()


    def process_matching(self):
        # get lokasi excel
        print("Match Process")
        self.file_result = FileResult()

        self.processed_provider = self.list_provider.get_provider_list().pop()
        self.processed_provider.save_perbandingan_model()

        # Compare and save the Item !
        pd.options.display.max_colwidth = None
        self.golden_record.set_golden_record_list_item()
        item_master_list = self.master_data.get_list_item_master_provider()



        for item_provider in self.processed_provider.get_list_item_provider():
            try:
                self.compare_item_provider_name_to_master_name(item_provider, item_master_list)
                # # # IF ITEM PROVIDER IS IN MASTER RECORD, DO NOT PROCESS IT AGAIN TO SHORTEN TIME


                # # # IF ITEM PROVIDER IS NOT IN MASTER RECORD
                # # # THEN PREDICT THE ITEM PROVIDER  SO WE CAN GET THE MASTER NAME
                self.predict_item_provider(item_provider)

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

    def process_matching_versus(self, provider):
        # get lokasi excel
        print("Match Process")
        self.set_dataset()
        self.file_result = FileResult()

        self.processed_provider = provider
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
                        # item_provider.set_golden_record(1)
                        item_provider.set_saved_in_golden_record(True)
                        item_provider.set_compared(True)
                        break

                if item_provider.get_golden_record_status() == "" and item_provider.get_saved_in_golden_record() is False:
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
                        alamat_ratio = fuzz.ratio(item_master.get_alamat_master(), item_provider.get_alamat())
                        total_score = float("{:.2f}".format((score + alamat_ratio) / 2))

                        if item_provider.get_total_score() <= total_score:
                            item_provider.set_total_score(total_score)
                            item_provider.set_id_model(self.processed_provider.get_primary_key_provider())
                            item_provider.set_label_name(item_master.get_nama_master())
                            item_provider.set_proba_score(nil)
                            item_provider.set_ratio(score)
                            item_provider.set_alamat_ratio(alamat_ratio)
                            item_provider.set_count_label_name(0)
                            item_provider.set_alamat_prediction(item_master.get_alamat_master())
                            item_provider.set_golden_record(0)
                            item_provider.set_saved_in_golden_record(False)
                            item_provider.set_status_item_provider("Direct 1")
                            item_provider.set_id_master(item_master.get_id_master())
                            item_provider.set_nama_master_provider(item_master.get_nama_master())
                            item_provider.set_alamat_master_provider(item_master.get_alamat_master())
                            item_provider.save()

                for item_master in self.master_data.get_list_item_master_provider():
                    if item_provider.get_total_score() <= 70 and item_provider.get_ratio() <= 90:
                        score = fuzz.ratio(item_provider.get_nama_provider(), item_master.get_nama_master())
                        # alamat_ratio = fuzz.ratio(item_master.get_alamat_master(), item_provider.get_alamat())
                        # total_score = float("{:.2f}".format((score + alamat_ratio) / 2))

                        if item_provider.get_ratio() <= score:
                            # item_provider.set_total_score(total_score)
                            item_provider.set_id_model(self.processed_provider.get_primary_key_provider())
                            item_provider.set_label_name(item_master.get_nama_master())
                            item_provider.set_proba_score(nil)
                            item_provider.set_ratio(score)
                            # item_provider.set_alamat_ratio(alamat_ratio)
                            item_provider.set_count_label_name(0)
                            item_provider.set_alamat_prediction(item_master.get_alamat_master())
                            item_provider.set_golden_record(0)
                            item_provider.set_saved_in_golden_record(False)
                            item_provider.set_status_item_provider("Direct 2")
                            item_provider.set_id_master(item_master.get_id_master())
                            item_provider.set_nama_master_provider(item_master.get_nama_master())
                            item_provider.set_alamat_master_provider(item_master.get_alamat_master())
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
        print("Create File Result")
        # # get nama asuransi
        nama_asuransi = self.file_result.get_nama()
        print("NAMA ASURANSI")
        print(nama_asuransi)
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
        if self.id is not None:
            self.id += 1
        self.id_file_result = self.file_result.pk
        self.id_model = self.processed_provider.get_primary_key_provider()
        self.match_percentage = 1.00
        self.save(force_insert=True)

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
                        # golden.save()

        pass


class Provider(models.Model):
    print("Provider")
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
            state_name = row.Provinsi
            city_name = row.Kota

            provider_object.set_provider_name(nama)
            provider_object.set_alamat(alamat)
            provider_object.set_ri(rawat_inap)
            provider_object.set_rj(rawat_jalan)
            provider_object.set_label_name("-")
            provider_object.set_proba_score(0)
            provider_object.set_count_label_name(0)
            provider_object.set_nama_alamat()
            provider_object.set_saved_in_golden_record(False)

            url = 'https://www.asateknologi.id/api/stateId'
            myobj = {'stateName': state_name, 'cityName': city_name}
            try:
                x = requests.post(url, json=myobj)
                state_id = x.json()["state_id"]
                city_id = x.json()["city_id"]

            except Exception as e:
                state_id = 0
                city_id = 0

            provider_object.set_city_id(city_id)
            provider_object.set_state_id(state_id)

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
