class ItemState:
    def __init__(self,id_state=None,state_name=None):
        self.id = id_state
        self.state_name = state_name
        self.city_list = []

    def set_state_id(self,id_state):
        self.id = id_state

    def set_state_name(self,state_name):
        self.state_name = state_name

    def set_city_list(self,city_list):
        self.city_list = city_list

    def get_city_list(self):
        return self.city_list

    def get_state_id(self):
        return self.id

    def get_state_name(self):
        return self.state_name
