class MasterData:
    def __init__(self,id=None,nama_provider=None,alamat=None,category_1=None,category_2=None,phone=None,
                 state_id=None,city_id=None):
        self.id = str(id)
        self.state_id = str(state_id)
        self.city_id = str(city_id)
        self.category_1 = str(category_1)
        self.category_2 = str(category_2)
        self.nama_provider = str(nama_provider).replace("_x000D_","")
        self.alamat = str(alamat).replace("_x000D_","")
        self.phone = str(phone)
        self.match_name = False
        self.match_address = False

    def set_alamat_master(self,alamat):
        self.alamat = alamat


    def set_match_address_true(self):
        self.match_address = True

    def get_match_address_boolean(self):
        return self.match_address
    def set_match_name_true(self):
        self.match_name = True

    def get_match_name_boolean(self):
        return self.match_name
    def set_id_master(self,id):
        self.id = id

    def set_nama_master(self,nama_provider):
        self.nama_provider = str(nama_provider).replace("_x000D_","")

    def get_id_master(self):
        return self.id

    def get_nama_master(self):
        return self.nama_provider


    def set_varian(self,varian):
        self.varian = varian


    def get_varian(self):
        return self.varian

    def get_alamat_master(self):
        return self.alamat
