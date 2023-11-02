import pandas as pd

from classM.ItemMaster import ItemMaster
from classM.Pembersih import Pembersih


class MasterData:
    def __init__(self):
        self.description = None
        self.lokasi_excel = "master_provider.xlsx"
        self.df = pd.read_excel(self.lokasi_excel)

        # clean the dataframe
        pembersih = Pembersih(self.df)

        self.dataframe_master = pembersih._return_df()
        self.list_item_master_provider = []
        self.dict_item_master_provider = {}
        self.set_list_item_master_provider()


    def get_raw_master(self):
        return self.df

    def get_master_excel_location(self):
        return self.lokasi_excel

    def add_to_list_master_provider(self,item):
        self.list_item_master_provider.append(item)

    def set_new_datafarame(self,df):
        self.clear()
        self.dataframe_master = None
        # clean the dataframe
        pembersih = Pembersih(df)
        self.dataframe_master = pembersih._return_df()
        self.set_list_item_master_provider()

    def set_list_item_master_provider(self):
        print("Create provider item list")
        for row in self.dataframe_master.itertuples(index=True, name='Sheet1'):


            master_provider_id = str(row.ProviderId)
            master_state_id = row.stateId
            master_city_id = row.cityId
            master_category_1 = row.Category_1
            master_category_2 = row.Category_2
            master_nama_provider = row.PROVIDER_NAME
            master_alamat = row.ADDRESS
            master_tel = row.TEL_NO

            try:
                itemMaster = ItemMaster(master_provider_id,
                                        master_state_id,
                                        master_city_id,
                                        master_category_1,
                                        master_category_2,
                                        master_nama_provider,
                                        master_alamat,
                                        master_tel)
                self.list_item_master_provider.append(itemMaster)
                self.dict_item_master_provider[master_provider_id] = itemMaster
            except:
                pass

        print("Finish Create provider item list")

    def get_dict_item_master_provider(self):
        return self.dict_item_master_provider

    def get_list_item_master_provider(self):
        return self.list_item_master_provider

    def clear(self):
        self.list_item_master_provider = []
        self.dict_item_master_provider = {}
    def set_description(self,description):
        self.description=description

    def get_description(self):
        return self.description


    def get_list_item_master_provider_json(self):
        ls = []
        for item_master in self.get_list_item_master_provider():
            ls.append(item_master.__dict__)
        return ls

    def get_dataframe(self):
        return self.dataframe_master
