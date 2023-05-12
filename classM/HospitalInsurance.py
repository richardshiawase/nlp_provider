import requests

from classM.ItemHospitalInsurance import ItemHospitalInsurance


class HospitalInsurance:
    def __init__(self, id_asuransi):
        self.item_hosins_list = []
        self.item_hosins_dict = {}
        url = 'https://www.asateknologi.id/api/inshos-by-insurance'
        myobj = {'id_asuransi': id_asuransi}
        x = requests.post(url, json=myobj)

        for val in x.json()["val"]:
            item = ItemHospitalInsurance()

            item.set_category(val["category_1"])
            item.set_hospital_name(val['provider_name'])
            item.set_hospital_address(val['address'])
            item.set_id_hosins(val["id"])
            item.set_hospital_id(val["HospitalId"])
            item.set_insurance_id(val["InsuranceId"])
            item.set_ri(val["Inpatient"])
            item.set_rj(val["Outpatient"])

            self.item_hosins_dict[item.get_id_hosins()] = item.__dict__
            self.item_hosins_list.append(item.__dict__)

    def get_item_hosins_list(self):
        return self.item_hosins_dict
