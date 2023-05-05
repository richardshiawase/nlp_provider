class ItemCity:
    def __init__(self,city_id=None,city_name=None,stateId=None):
        self.id = city_id
        self.city_name = city_name
        self.state_id = stateId

    def set_city_id(self,city_id):
        self.id = city_id

    def set_city_name(self,city_name):
        self.city_name = city_name

    def set_state_id(self,state_id):
        self.state_id = state_id

    def get_city_id(self):
        return self.id

    def get_city_name(self):
        return self.city_name

    def get_state_id(self):
        return self.state_id