class ItemMaster:
    def __init__(self,provider_id,stateId,cityId,category_1,category_2,provider_name,address,tel_no):
        self.provider_id = provider_id
        self.stateId = stateId
        self.cityId = cityId
        self.category_1 = category_1
        self.category_2 = category_2
        self.provider_name = str(provider_name).replace("_x000D_", "")
        self.address = str(address).replace("_x000D_", "")
        self.tel_no = tel_no


    def set_alamat_master(self, alamat):
        self.address = alamat

    def set_match_address_true(self):
        self.match_address = True

    def get_match_address_boolean(self):
        return self.match_address

    def set_match_name_true(self):
        self.match_name = True

    def get_match_name_boolean(self):
        return self.match_name

    def set_id_master(self, id):
        self.provider_id = id

    def set_nama_master(self, nama_provider):
        self.nama_provider = str(nama_provider).replace("_x000D_", "")

    def get_id_master(self):
        return self.provider_id

    def get_nama_master(self):
        return self.provider_name.strip()

    def set_varian(self, varian):
        self.varian = varian

    def get_varian(self):
        return self.varian

    def get_alamat_master(self):
        return self.address

    def set_telepon_master(self, tel):
        self.tel_no = tel

    def get_telepon_master(self):
        return self.tel_no

    def set_category_1_master(self, cat):
        self.category_1 = cat

    def get_category_1_master(self):
        return self.category_1

    def set_category_2_master(self, cat):
        self.category_2 = cat

    def get_category_2_master(self):
        return self.category_2

    def set_state_id_master(self, id):
        self.stateId = id

    def get_state_id_master(self):
        return self.stateId
