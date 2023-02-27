import pathlib


class FilePembandingAsuransi:
    def __init__(self):
        pass

    def set_provider_list(self,provider_data_list):
        self.provider_data_list = provider_data_list

    def get_provider_list(self):
        return self.provider_data_list

    def set_nama_file_pembanding(self):
        # self.nama_file = pathlib.Path("media/"+self.uploaded_file.name)
        self.nama_file = self.uploaded_file.name

    def set_lokasi_file_pembanding(self):
        self.lokasi_file_pembanding = self.perbandingan_model.file_location


    def set_extension_file_pembanding(self):
        try:
            self.file_extension = pathlib.Path("media/"+self.uploaded_file).suffix
        except:
            self.file_extension = pathlib.Path("media/" +self.uploaded_file.name).suffix
    def set_nama_asuransi(self,nama_asuransi):
        print("nama asuransi "+nama_asuransi)
        self.nama_asuransi = nama_asuransi

    def set_nama_asuransi_dari_model(self):
        self.nama_asuransi = self.get_perbandingan_model().nama_asuransi

    def set_uploaded_file(self,w):
        self.uploaded_file = w
        try:
            self.set_nama_file_pembanding()
        except:
            print("Tidak ada nama file pembanding")

        try:
            self.set_extension_file_pembanding()
        except:
            print("Tidak ada extension")
    def set_perbandingan_model(self,perbandingan_model):
        self.perbandingan_model = perbandingan_model


    def get_perbandingan_model(self):
        return self.perbandingan_model

    def get_lokasi_pembanding(self):
        print("lokasi file "+self.lokasi_file_pembanding)
        return self.lokasi_file_pembanding
    def get_uploaded_file(self):
        return self.uploaded_file

    def get_nama_asuransi(self):
        return self.nama_asuransi

    def get_extension_file_pembanding(self):
        return self.file_extension

    def get_nama_file_pembanding(self):
        return self.nama_file

    def create_hasil_perbandingan_file(self):
        pass
