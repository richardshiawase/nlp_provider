class ItemPembanding:
    def __init__(self,nama_provider,alamat,label_name,proba_score,count_label,ri,rj):
        self.nama_provider = str(nama_provider).strip()
        self.alamat = str(alamat).strip()
        self.label_name = str(label_name).strip()
        self.proba_score = str(proba_score).strip()
        self.count_label_name = str(count_label).strip()
        self.ri = str(ri).strip()
        self.rj = str(rj).strip()

    def get_ri(self):
        if(self.ri == "Y"):
            return "1"
        elif self.ri == "N":
            return "0"

    def get_rj(self):
        if (self.rj == "Y"):
            return "1"
        elif self.rj == "N":
            return "0"


    def set_alamat_prediction(self,alamat_prediction):
        self.alamat_prediction = str(alamat_prediction)

    def get_alamat_prediction(self):
        return self.alamat_prediction

    def set_mapped_times(self,times):
        self.mapped_times = times

    def set_id_master(self,id):
        self.id_master = id


    def get_id_master(self):
        return self.id_master

    def get_mapped_times(self):
        return self.mapped_times

    def set_count_label_name(self,count):
        self.count_label_name = count

    def set_selected(self,selected):
        self.selected = selected

    def set_nama_asuransi(self,nama_asuransi):
        self.nama_asuransi = nama_asuransi


    def get_count_label_name(self):
        return self.count_label_name

    def get_nama_asuransi(self):
        return self.nama_asuransi

    def get_nama_provider(self):
        return self.nama_provider

    def get_alamat(self):
        return self.alamat

    def get_label_name(self):
        return self.label_name

    def get_proba_score(self):
        return self.proba_score

    def get_selected(self):
        return self.selected
