class ItemAsuransi:
    def __init__(self,id_asuransi = None,singkatan = None,nama_lengkap_asuransi=None):
        self.id_asuransi = id_asuransi
        self.singkatan = singkatan
        self.nama_lengkap_asuransi = nama_lengkap_asuransi
        self.hospital_linked_list = []

    def set_id_asuransi(self, id_asuransi):
        self.id_asuransi = id_asuransi

    def get_id_asuransi(self):
        return self.id_asuransi

    def set_nama_lengkap_asuransi(self, nama_asuransi):
        self.nama_lengkap_asuransi = nama_asuransi

    def get_nama_lengkap_asuransi(self):
        return self.nama_lengkap_asuransi

    def set_singkatan_asuransi(self,singkatan_asuransi):
        self.singkatan = singkatan_asuransi

    def get_singkatan_asuransi(self):
        return self.singkatan

    def set_hospital_linked_list(self,item_master_list):
        self.hospital_linked_list = item_master_list
        self.linked_hospital_count = len(self.hospital_linked_list)

    def get_hospital_linked_list(self):
        return self.hospital_linked_list

    def set_processed(self,bool):
        self.current = bool

    def get_processed(self):
        return self.current

