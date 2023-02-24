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
