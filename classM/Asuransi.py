import requests

from classM.HospitalInsurance import HospitalInsurance
from classM.ItemAsuransi import ItemAsuransi
from classM.MasterData import MasterData


class Asuransi:
    def __init__(self):
        self.item_asuransi_dict = {}
        self.item_asuransi_list = []
        url = 'https://www.asateknologi.id/api/insuranceall'
        x = requests.get(url)
        for val in x.json()["val"]:
            item_asuransi = ItemAsuransi()
            item_asuransi.set_id_asuransi(val["id_asuransi"])
            item_asuransi.set_singkatan_asuransi(val["nama_asuransi"])
            self.item_asuransi_list.append(item_asuransi.__dict__)
            hosins = HospitalInsurance(item_asuransi.get_id_asuransi())
            item_asuransi.set_hospital_linked_list(hosins.get_item_hosins_list())
            self.item_asuransi_dict[item_asuransi.get_singkatan_asuransi()] = item_asuransi.__dict__


    def get_dict_item_asuransi(self):
        return self.item_asuransi_dict

    def get_list_item_asuransi(self):
        return self.item_asuransi_list

    def reload(self):
        pass


