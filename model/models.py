from django.db import models

# Create your models here.
class Provider_Model(models.Model):
    model_name = models.CharField(max_length=30)
    accuracy_score = models.DecimalField(max_digits=5,decimal_places=2)
    model_location = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"{self.model_name} memiliki {self.accuracy_score} akurasi, dibuat pada {self.created_at}."


class Perbandingan(models.Model):
    nama_asuransi = models.CharField(max_length=500)
    match_percentage = models.DecimalField(max_digits=5,decimal_places=2)
    status_finish = models.CharField(max_length=8)
    file_location = models.CharField(max_length=1000)
    file_location_result = models.CharField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)




    def say_hello(self):
        print("Hello Guys")

    @staticmethod
    def get_model_from_filter(nama_asuransi):
        mydata = Perbandingan.objects.filter(nama_asuransi__contains=nama_asuransi).order_by('created_at').first()
        if not mydata:
            return False
        return mydata

    def get_nama_asuransi_model(self):
        return self.nama_asuransi

    def get_lokasi_excel_pembanding(self):
        return self.file_location



class Provider_Perbandingan(models.Model):
    nama_asuransi = models.CharField(max_length=500)
    perbandingan_id = models.CharField(max_length=2)
    name = models.CharField(max_length=500)
    address = models.CharField(max_length=1000)
    selected = models.CharField(max_length=2)

    def __str__(self):
        return f" perbandingan id {self.nama_asuransi} "
