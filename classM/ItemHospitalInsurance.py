class ItemHospitalInsurance:
    def __init__(self):
        pass

    def set_category(self,category):
        if(category == "1"):
            self.hospital_category = "RS"
        elif (category == "2"):
            self.hospital_category = "KLINIK"
        elif(category == "3"):
            self.hospital_category = "APOTIK"
        elif(category == "4"):
            self.hospital_category = "LAB"
        elif(category == "5"):
            self.hospital_category = "OPTIK"
        else:
            self.hospital_category = "-"


    def get_category(self):
        return self.hospital_category

    def set_hospital_name(self,name):
        self.hospital_name = name

    def get_hospital_name(self):
        return self.hospital_name

    def set_hospital_address(self,address):
        self.hospital_address = address

    def get_hospital_address(self):
        return self.hospital_address

    def set_id_hosins(self,id):
        self.id_hosins = id

    def get_id_hosins(self):
        return self.id_hosins

    def set_hospital_id(self,hospital_id):
        self.hospital_id = hospital_id

    def get_hospital_id(self):
        return self.hospital_id

    def set_insurance_id(self,insurance_id):
        self.insurance_id = insurance_id

    def set_ri(self,ri):
        self.ri = str(ri)

    def get_ri(self):
        return self.ri

    def set_rj(self,rj):
        self.rj = str(rj)

    def get_rj(self):
        return self.rj

