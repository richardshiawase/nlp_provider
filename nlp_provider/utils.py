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
        return self.ri

    def get_rj(self):
        return self.rj

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

class Prediction:
    def __init__(self,nama_provider,alamat):
        self.nama_provider = nama_provider
        self.alamat = alamat

    def get_nama_provider(self):
        return self.nama_provider

    def get_count_refer(self):
        return self.count_refer

    def get_alamat(self):
        return self.alamat

    def set_count_refer(self,count_refer):
        self.count_refer = count_refer



class MasterData:
    def __init__(self,id,nama_provider,alamat,category_1,category_2,phone,state_id,city_id):
        self.id = str(id)
        self.state_id = str(state_id)
        self.city_id = str(city_id)
        self.category_1 = str(category_1)
        self.category_2 = str(category_2)
        self.nama_provider = str(nama_provider).replace("_x000D_","")
        self.alamat = str(alamat).replace("_x000D_","")
        self.phone = str(phone)


    def set_varian(self,varian):
        self.varian = varian


    def get_varian(self):
        return self.varian


class PredictionId:
    def __init__(self,prediction,id_master):
        self.prediction = prediction
        self.id_master = id_master

class FilePembandingAsuransi:
    def __init__(self,nama_asuransi,match_percentage,status_finish,file_location,file_location_result,created_at):
        self.nama_asuransi = nama_asuransi
        self.match_percentage = match_percentage
        self.status_finish = status_finish
        self.file_location = file_location
        self.file_location_result = file_location_result
        self.created_at = created_at

    def set_provider_list(self,provider_data_list):
        self.provider_data_list = provider_data_list

    def get_provider_list(self):
        return self.provider_data_list
