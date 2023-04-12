from django.db import models
from django.db import connections
from django.forms import model_to_dict


# Create your models here.
class List_Processed_Provider(models.Model):
    id_provider = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_provider_list(self,provider_list):
        self.provider_list = []
        self.provider_list = provider_list

    def get_provider_list(self):
        return self.provider_list

    def get_a_provider_from_id(self,pk):
        id_provider = pk
        for data in self.get_provider_list():
            if data.get_primary_key_provider() == str(id_provider):
                return data


class Provider_Model(models.Model):
    model_name = models.CharField(max_length=30)
    accuracy_score = models.DecimalField(max_digits=5, decimal_places=2)
    model_location = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

class ItemProvider(models.Model):
    id_model = models.CharField(max_length=500)
    id_asuransi = models.CharField(max_length=500)
    nama_provider = models.CharField(max_length=500)
    alamat = models.CharField(max_length=500)
    label_name = models.CharField(max_length=300)
    proba_score = models.CharField(max_length=10)
    ratio = models.CharField(max_length=10)
    alamat_ratio = models.CharField(max_length=10)
    total_score = models.CharField(max_length=10)
    count_label_name = models.CharField(max_length=2)
    ri = models.CharField(max_length=2)
    rj = models.CharField(max_length=2)
    nama_alamat = models.CharField(max_length=500)
    alamat_prediction = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_status(self,status):
        self.status = status

    def get_status(self):
        return self.status

    def set_processed(self, found):
        self.found = found

    def is_processed(self):
        return self.found

    def set_alamat_ratio(self,alamat_ratio):
        self.alamat_ratio = alamat_ratio

    def get_alamat_ratio(self):
        return self.alamat_ratio

    def set_ratio(self,ratio):
        self.ratio = ratio

    def get_ratio(self):
        return self.ratio

    def set_total_score(self,total_score):
        self.total_score = total_score

    def get_total_score(self):
        return self.total_score

    def set_id(self,pk):
        self.id = pk

    def get_id(self):
        return self.id

    def set_id_model(self,id_model):
        self.id_model = id_model

    def get_id_model(self):
        return self.id_model

    def get_ri(self):
        return self.ri

    def get_rj(self):
        return self.rj

    def set_alamat_prediction(self, alamat_prediction):
        self.alamat_prediction = str(alamat_prediction)

    def get_alamat_prediction(self):
        return self.alamat_prediction

    def set_mapped_times(self, times):
        self.mapped_times = times

    def get_mapped_times(self):
        return self.mapped_times

    def set_count_label_name(self, count):
        self.count_label_name = count

    def set_selected(self, selected):
        self.selected = selected

    def set_nama_asuransi(self, nama_asuransi):
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
        return self.label_name.strip()

    def get_proba_score(self):
        return float(self.proba_score)

    def get_selected(self):
        return self.selected

    def save_item_provider(self):
        print("save item providewr")
        # # # Save Perbandingan Model

        self.save()

    def set_provider_name(self, value):
        self.nama_provider = value

    def set_alamat(self, param):
        self.alamat = param

    def set_nama_alamat(self):
        self.nama_alamat = self.get_nama_provider() + "#" + self.get_alamat()

    def get_nama_alamat(self):
        return self.nama_alamat

    def set_label_name(self, y_preds):
        self.label_name = y_preds

    def set_proba_score(self, nil):
        self.proba_score = nil

    def set_ri(self, param):
        if param.lower() == "y":
            self.ri = 1
        if param.lower() == "n":
            self.ri = 0

    def set_rj(self, param):
        if param.lower() == "y":
            self.rj = 1
        if param.lower() == "n":
            self.rj = 0

    def set_id_asuransi(self, param):
        self.id_asuransi = param

    def get_id_asuransi(self):
        return self.id_asuransi

    def set_validity(self,valid):
        self.validity = valid

    def is_valid(self):
        return self.validity




class Provider(models.Model):
    nama_asuransi = models.CharField(max_length=500)
    match_percentage = models.DecimalField(max_digits=5, decimal_places=2)
    status_finish = models.CharField(max_length=8)
    file_location = models.CharField(max_length=1000)
    file_location_result = models.CharField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)

    @staticmethod
    def get_model_from_filter(nama_asuransi):
        mydata = Provider.objects.filter(nama_asuransi__contains=nama_asuransi).order_by('created_at').first()
        if not mydata:
            return False
        return mydata

    def get_created_at(self):
        return self.created_at

    def get_nama_asuransi_model(self):
        return self.nama_asuransi

    def set_nama_asuransi_model(self, nama_asuransi):
        self.nama_asuransi = nama_asuransi

    def set_id_asuransi_model(self, id_asuransi):
        self.id_asuransi = id_asuransi

    def get_id_asuransi_model(self):
        return self.id_asuransi

    def set_id(self,pk):
        self.id = pk

    def get_file_location_result(self):
        return "media" + self.file_location_result

    def get_lokasi_excel_pembanding(self):
        print("Get Excel Pembanding")
        return self.file_location

    def set_file_location(self, file_location):
        self.file_location = file_location

    def get_primary_key_provider(self):
        return str(self.pk)

    def save_perbandingan_model(self):
        # # # Save Perbandingan Model
        nama_asuransi = self.get_nama_asuransi_model()
        self.nama_asuransi = nama_asuransi
        self.match_percentage = 0.00
        self.file_location_result = "/" + nama_asuransi + "_result.xlsx"
        self.save()

    def set_list_item_provider(self, list_item_provider):
        self.list_item_provider = []
        self.list_item_provider = list_item_provider

    def get_list_item_provider(self):
        return self.list_item_provider

    def get_list_item_provider_json(self):
        ls = []
        for item_provider in self.get_list_item_provider():
            del item_provider._state
            ls.append(item_provider.__dict__)
        return ls

class Dataset(models.Model):
    course_title = models.CharField(max_length=500)
    alamat = models.CharField(max_length=500)
    subject = models.CharField(max_length=500)
    master_address = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)