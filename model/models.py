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


    def __str__(self):
        return f"{self.nama_asuransi} dengan persentase match {self.match_percentage} dengan status selesai {self.status_finish} lokasi perbandingan {self.file_location} dan lokasi result {self.file_location_result}"


class Provider_Perbandingan(models.Model):
    nama_asuransi = models.CharField(max_length=500)
    perbandingan_id = models.CharField(max_length=2)
    name = models.CharField(max_length=500)
    address = models.CharField(max_length=1000)
    selected = models.CharField(max_length=2)

    def __str__(self):
        return f" perbandingan id {self.nama_asuransi} "
