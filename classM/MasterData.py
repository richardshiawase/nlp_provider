import pandas as pd

from classM.ItemMaster import ItemMaster
from classM.Pembersih import Pembersih


class MasterData:
    def __init__(self):
        self.lokasi_excel = "master_provider.xlsx"
        df = pd.read_excel(self.lokasi_excel)

        # clean the dataframe
        pembersih = Pembersih(df)

        self.dataframe_master = pembersih._return_df()
        self.list_item_master_provider = []
        self.set_list_item_master_provider()

    def get_master_excel_location(self):
        return self.lokasi_excel

    def add_to_list_master_provider(self,item):
        self.list_item_master_provider.append(item)

    def set_list_item_master_provider(self):
        print("Create provider item list")
        print(self.dataframe_master["Category_1"])
        # self.dataframe_master["Category_1"] = self.dataframe_master["Category_1"]
        # self.dataframe_master["Category_1"] = pd.to_numeric(self.dataframe_master["Category_1"])
        for row in self.dataframe_master.itertuples(index=True, name='Sheet1'):

            master_provider_id = row.ProviderId
            master_state_id = row.stateId
            master_city_id = row.cityId
            master_category_1 = row.Category_1
            master_category_2 = row.Category_2
            master_nama_provider = row.PROVIDER_NAME
            master_alamat = row.ADDRESS
            master_tel = row.TEL_NO

            itemMaster = ItemMaster(master_provider_id,
                                    master_state_id,
                                    master_city_id,
                                    master_category_1,
                                    master_category_2,
                                    master_nama_provider,
                                    master_alamat,
                                    master_tel)
            self.list_item_master_provider.append(itemMaster)


    def get_list_item_master_provider(self):
        return self.list_item_master_provider

    def get_list_item_master_provider_json(self):
        ls = []
        for item_master in self.get_list_item_master_provider():
            ls.append(item_master.__dict__)
        return ls

    def get_dataframe(self):
        return self.dataframe_master
