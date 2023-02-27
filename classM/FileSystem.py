from django.core.files.storage import FileSystemStorage

import django
django.setup()
from model import models
from model.models import Perbandingan


class FileSystem:
    def __init__(self,filePembandingAsuransi):
        self.file = filePembandingAsuransi
        self.fst = FileSystemStorage()

    def get_path(self):
        return self.fst.path(self.file.get_nama_file_pembanding())


    def save_file(self):
        uploaded_file_name = self.file.get_nama_file_pembanding()
        uploaded_file = self.file.get_uploaded_file()
        if self.allowed_extension() is True:
            self.saved_file = self.fst.save(uploaded_file_name,uploaded_file)
            self.is_the_insurance_ever_compared()
            return True
        else:
            return False

    def get_saved_file(self):
        try:
            return "media/"+self.saved_file
        except:
            return self.get_file_loc_result()
    def allowed_extension(self):
        extension = self.file.get_extension_file_pembanding()
        allowed = True if extension == ".xlsx" else False
        return allowed

    def get_perbandingan_model(self):
        return self.file.get_perbandingan_model()
    def get_nama_asuransi(self):
        return self.file.get_nama_asuransi()

    def get_file_loc_result(self):
        return "media/"+self.get_perbandingan_model().file_location_result

    def save_perbandingan_to_dashboard(self):
        pass

    def get_lokasi_file_pembanding(self):
        return self.file.get_lokasi_pembanding()

    def set_perbandingan_model(self,mydata):
        self.file.set_perbandingan_model(mydata)
        self.file.set_lokasi_file_pembanding()
        self.file.set_nama_asuransi_dari_model()

    def get_perbandingan_from_nama_asuransi(self,nama_asuransi):
        mydata = Perbandingan.objects.filter(nama_asuransi__contains=nama_asuransi).order_by('created_at').first()
        if not mydata:
            return False
        return mydata

    def is_the_insurance_ever_compared(self):
        nama_asuransi = self.file.get_nama_asuransi()
        mydata = Perbandingan.objects.filter(nama_asuransi__contains=nama_asuransi).order_by('created_at').values()

        if not mydata:
            print("ATAS")
            perbandingan_model = models.Perbandingan(nama_asuransi=nama_asuransi,
                                                     match_percentage=0,
                                                     status_finish="PROCESSING", file_location=self.get_path())

        else:
            print("BAWAH")
            perbandingan_model = Perbandingan.objects.filter(pk=mydata[0]["id"]).values()

        self.file.set_perbandingan_model(perbandingan_model)

    def save_perbandingan_model(self):
        # # # Save Perbandingan Model
        perbandingan_model = self.get_perbandingan_model()
        nama_asuransi = self.get_nama_asuransi()
        perbandingan_model.file_location_result = "/" + nama_asuransi + "_result.xlsx"
        perbandingan_model.save()
