from django.db import models
from django.db import connections
# Create your models here.
class List_Processed_Provider(models.Model):
    id_provider = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

class Provider_Model(models.Model):
    model_name = models.CharField(max_length=30)
    accuracy_score = models.DecimalField(max_digits=5,decimal_places=2)
    model_location = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)





class ItemProvider(models.Model):
    id_asuransi = models.CharField(max_length=500)
    nama_provider = models.CharField(max_length=500)
    alamat = models.CharField(max_length=500)
    label_name = models.CharField(max_length=300)
    proba_score = models.CharField(max_length=10)
    count_label_name = models.CharField(max_length=2)
    ri = models.CharField(max_length=2)
    rj = models.CharField(max_length=2)


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

    def save_item_provider(self):
        print("save item providewr")
        # # # Save Perbandingan Model

        self.save()

    def set_provider_name(self, value):
        self.nama_provider = value

    def set_alamat(self, param):
        self.alamat = param


    def set_label_name(self, y_preds):
        self.label_name = y_preds

    def set_proba_score(self, nil):
        self.proba_score = nil

    def set_ri(self, param):
        self.ri = param
        pass

    def set_rj(self, param):
        self.rj = param
        pass

    def set_id_asuransi(self, param):
        self.id_asuransi = param
        pass


class Provider(models.Model):
    nama_asuransi = models.CharField(max_length=500)
    match_percentage = models.DecimalField(max_digits=5,decimal_places=2)
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

    def get_nama_asuransi_model(self):
        return self.nama_asuransi

    def set_nama_asuransi_model(self,nama_asuransi):
        self.nama_asuransi = nama_asuransi

    def get_file_location_result(self):
        return "media"+self.file_location_result

    def get_lokasi_excel_pembanding(self):
        print("Get Excel Pembanding")
        return self.file_location

    def set_file_location(self,file_location):
        self.file_location = file_location

    def save_perbandingan_model(self):
        # # # Save Perbandingan Model
        nama_asuransi = self.get_nama_asuransi_model()
        self.nama_asuransi = nama_asuransi
        self.match_percentage = 0.00
        self.file_location_result = "/" + nama_asuransi + "_result.xlsx"
        self.save()

    def set_list_item_provider(self,list_item_provider):
        self.list_item_provider = list_item_provider

    def get_list_item_provider(self):
        return self.list_item_provider


