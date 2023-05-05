import requests

from classM.Cities import Cities
from classM.ItemState import ItemState


class States:
    def __init__(self):
        url = 'https://www.asateknologi.id/api/stateall'
        self.state_dict = {}
        x = requests.get(url)
        self.city = Cities()

        for val in x.json()["val"]:
            item_state = ItemState()
            item_state.set_state_name(val["StateName"])
            item_state.set_state_id(val["id"])

            if item_state.get_state_id() in self.city.get_item_city_state_dict():
                item_state.set_city_list(self.city.get_item_city_state_dict()[item_state.get_state_id()])
            self.state_dict[item_state.get_state_name()] = item_state


    def get_city(self):
        return self.city

    def set_item_state_dict(self, state_list):
        self.state_dict = state_list

    def get_item_state_dict(self):
        return self.state_dict

    def get_item_city_list(self):
        ls = []
        for city_list in self.city.get_item_city_state_dict().values():
             ls.extend(city_list)
        return ls
