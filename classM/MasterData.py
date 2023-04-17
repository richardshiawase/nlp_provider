import pandas as pd

from classM.ItemMaster import ItemMaster
from classM.Pembersih import Pembersih


class MasterData:
    def __init__(self,lokasi_excel):
        self.lokasi_excel = lokasi_excel
        df = pd.read_excel(lokasi_excel)
        # clean the dataframe
        pembersih = Pembersih(df)

        self.dataframe_master = pembersih._return_df()
        self.list_item_master_provider = []
        self.set_list_item_master_provider()

    def get_master_excel_location(self):
        return self.lokasi_excel

    def set_list_item_master_provider(self):
        print("Create provider item list")

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

    def get_dataframe(self):
        return self.dataframe_master
