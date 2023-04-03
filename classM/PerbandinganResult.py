import pandas as pd

from classM.ColumnOutput import ColumnOutput
from classM.ExcelBacaTulis import ExcelBacaTulis
from django.db import connection


class PerbandinganResult():
    def __init__(self):
        self.column_output = ColumnOutput()
        self.ex = ExcelBacaTulis()

        pass

    def map_dataframe_to_column_output(self,dataframe):
        print("Map Dataframe To Column Output")
        result_column = self.column_output.get_column_to_output()
        mappeds = {}
        for x in result_column:
            try:
                mappeds[x] = dataframe[x]
            except Exception as e:
                print("Tidak ditemukan output column "+x+" di dataframe "+str(e))
                mappeds[x] = pd.Series([])

        return mappeds

    def create_file(self,df_handler):

        # get lokasi excel
        lokasi_excel = df_handler.perbandingan_model.get_lokasi_excel_pembanding()

        # read excel by lokasi excel and get the dataframe
        dataframe_pembanding = df_handler.convert_to_dataframe_from_excel(lokasi_excel)

        # header for initial process
        header = ['Nama', 'Alamat', 'Alamat_Prediction', 'RI', 'RJ']

       # create process excel and convert to dict
        proccesed_dict = df_handler.convert_dataframe_refer_to_specified_column(header,dataframe_pembanding)

        # process result file
        processed_dataframe = df_handler.process_result_file_to_dataframe(proccesed_dict)

        # # # map processed dataframe column to output desired column
        mapped = self.map_dataframe_to_column_output(processed_dataframe)

        # # # convert mapped list to dataframe
        df = pd.DataFrame(mapped)

        # # get nama asuransi
        nama_asuransi = df_handler.perbandingan_model.get_nama_asuransi_model()
        #
        # # write to excel
        self.ex.write_to_excel(nama_asuransi,"_result",df)
        #
        # # save perbandingan model
        df_handler.perbandingan_model.save_perbandingan_model()

    def create_file_result_with_id_master(self, df_handler):
        self.create_file(df_handler)

        lokasi_excel = "master_provider.xlsx"

        # read excel by lokasi excel and get the dataframe
        dataframe_pembanding = df_handler.convert_to_dataframe_from_excel(lokasi_excel)

        # set header to read
        header = ['ProviderId','PROVIDER_NAME','ADDRESS','stateId','cityId','Category_1']

        # create process excel and convert to dict
        proccesed_dict = df_handler.convert_dataframe_refer_to_specified_column(header, dataframe_pembanding)

        # # # # combine the result with id
        processed_dataframe = df_handler.process_result_id_master_to_dataframe(proccesed_dict)


        # # # map processed dataframe column to output desired colum
        header_output = ['IdMaster','Master_Nama','Master_Alamat', 'Nama', 'Alamat', 'Score', 'Compared', 'Clean', 'RI', 'RJ']
        self.column_output.set_column_output(header_output)
        mapped = self.map_dataframe_to_column_output(processed_dataframe)

        # # # convert mapped list to dataframe
        df = pd.DataFrame(mapped)

        # # # write to excel
        self.ex.write_to_excel(df_handler.perbandingan_model.get_nama_asuransi_model(),"_result_final",df)


        pass

