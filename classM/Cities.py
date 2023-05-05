import requests

from classM.ItemCity import ItemCity


class Cities:
    def __init__(self):
        url = 'https://www.asateknologi.id/api/cityall'
        self.city_state_dict = {}
        self.city_only_dict = {}
        x = requests.get(url)
        for val in x.json()["val"]:
            item_city = ItemCity()
            item_city.set_city_name(val["CityName"])
            item_city.set_state_id(val["StateId"])
            item_city.set_city_id(val["id"])
            if item_city.get_state_id() not in self.city_state_dict:
                self.city_state_dict[item_city.get_state_id()] = []

            self.city_state_dict[item_city.get_state_id()].append(item_city)
            self.city_only_dict[item_city.get_city_name()] = item_city

    def get_item_city_state_dict(self):
        return self.city_state_dict

    def get_item_city_only_dict(self):
        return self.city_only_dict
