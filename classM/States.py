import requests

from classM.ItemState import ItemState


class States:
    def __init__(self):
        url = 'https://www.asateknologi.id/api/stateall'
        self.state_list = []
        x = requests.get(url)
        for val in x.json()["val"]:
            item_state = ItemState()
            item_state.set_state_name(val["StateName"])
            item_state.set_state_id(val["id"])
            self.state_list.append(item_state)


    def set_item_state_list(self,state_list):
        self.state_list = state_list

    def get_item_state_list(self):
        return self.state_list
