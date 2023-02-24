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
