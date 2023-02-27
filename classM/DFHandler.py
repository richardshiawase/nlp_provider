import pickle
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

import warnings

from classM.ExcelBacaTulis import ExcelBacaTulis
from classM.ItemPembanding import ItemPembanding
from classM.Pembersih import Pembersih

warnings.simplefilter(action='ignore', category=FutureWarning)
class DFHandler:
    def __init__(self):
        filename = 'tfidf_vec.pickle'
        self.tfidf_vec1 = pickle.load(open(filename, 'rb'))
        filename = 'finalized_model.sav'
        self.loaded_model1 = pickle.load(open(filename, 'rb'))
        self.ex = ExcelBacaTulis()
        self.provider_list = []
        self.df = None

    def set_file_system(self,fs):
        self.file_system = fs

    def read_from_excel(self,excel):
        self.df = self.ex.baca_excel(excel)
        self.pembersih = Pembersih(self.df)

    def set_dataframe(self,dataframe):
        if dataframe is not None:
            self.df = dataframe

    def get_data_frame(self):
        df = self.pembersih._return_df()
        return df

    def cacah_dataframe(self,df):
        split_row_each = 800
        start_index = 0
        iteration_count = int(df.shape[0] / split_row_each)
        sisa = df.shape[0] % split_row_each
        sisa_row = iteration_count * split_row_each + sisa
        df_list = []
        for x in range(iteration_count):
            end_index = start_index + split_row_each
            df_new = df.iloc[start_index:end_index]
            start_index = end_index
            # df_list.append([df_new,lr])
            df_list.append(df_new)
        aw = lambda x, y: y if x > 0 else 0
        df_last = df.iloc[start_index:aw(sisa, sisa_row)]
        df_list.append(df_last)

        return df_list

    def set_df_dataset(self,df):
        self.df_dataset = df

    def vectorize_text(self,text, tfidf_vec):
        # text = "Klinik Ananda"
        my_vec = tfidf_vec.transform([text])
        return my_vec.toarray()

    def pool_process_df(self,df):
        # for df in df_list:
        # dataframe Name and Age columns
        pd.options.display.max_colwidth = None
        provider_name_list = []
        provider_name_predict_list = []
        score_list = []
        provider_object_list = []
        df_non_duplicate = self.df_dataset

        df_result = pd.DataFrame()
        for row in tqdm(df.itertuples(), total=df.shape[0]):
            new_string = row.nama
            alamat = row.alamat
            ri = row.RI
            rj = row.RJ

            # replace with df_nama_alamat
            nama_alamat = row.nama_alamat

            provider_name = new_string

            # course_title = apotik  klinik kimia farma  cilegon#jl. s.a. tirtayasa no 12

            # val = (df_non_duplicate['course_title'].str.lower().str.strip().eq(nama_alamat))
            val = (df_non_duplicate['course_title'].eq(nama_alamat))

            res = df_non_duplicate[val]

            provider_name_list.append(provider_name)
            # load the model from disk
            sample1 = self.vectorize_text(nama_alamat, self.tfidf_vec1)
            y_preds = self.loaded_model1.predict(sample1)
            p = self.loaded_model1.predict_proba(sample1)
            ix = p.argmax(1).item()
            nil = (f'{p[0, ix]:.2}')

            provider_name_predict_list.append(y_preds)
            score_list.append(nil)
            provider_object = ItemPembanding(provider_name, alamat, y_preds, nil, 0, ri, rj)

            if not res.empty:
                pred = str(y_preds).replace("[", "").replace("]", "").replace("'", "")
                val_master = (df_non_duplicate['subject'].eq(pred))
                res_master = df_non_duplicate[val_master]
                al = res_master["alamat"].head(1)
                try:
                    alamat_pred = al.values[0]
                    data_append = {
                        "Provider Name": provider_name,
                        "Alamat": alamat,
                        "Prediction": y_preds,
                        "Alamat Prediction": alamat_pred,
                        "Score": nil,
                        "Compared": 1,
                        "Clean": new_string,
                        "ri": ri,
                        "rj": rj
                    }
                    provider_object.set_alamat_prediction(alamat_pred)
                    df1 = pd.DataFrame(data_append)
                except:
                    print("error")


            elif res.empty:

                data_append = {
                    "Provider Name": provider_name,
                    "Alamat": alamat,
                    "Prediction": y_preds,
                    "Alamat Prediction": "-",
                    "Score": nil,
                    "Compared": 0,
                    "Clean": new_string,
                    "ri": ri,
                    "rj": rj
                }
                provider_object.set_alamat_prediction("-")
                df1 = pd.DataFrame(data_append)
            provider_object_list.append(provider_object)
            df_result = df_result.append(df1, ignore_index=True)
            # Provider_Perbandingan_data = models.Provider_Perbandingan(nama_asuransi=perbandingan_model.nama_asuransi,
            #                                                           perbandingan_id=1,
            #                                                           name=provider_name_label, address="-", selected=0)
            # Provider_Perbandingan_data.save()

        return df_result


    def pool_handler(self):
        df = self.get_data_frame()
        df_nama = df['Nama Provider']
        df_alamat = df['Alamat']
        df_ri = df['RI']
        df_rj = df['RJ']
        df_nama_alamat = df_nama.map(str) + '#' + df_alamat.map(str)
        df_lengkap = pd.DataFrame(
            {'nama': df_nama, 'alamat': df_alamat, 'RI': df_ri, 'RJ': df_rj, 'nama_alamat': df_nama_alamat})

        # # # Split dataframe to many
        df_list = self.cacah_dataframe(df_lengkap)

        # # # Using multiprocess with pool as many as dataframe list
        p = Pool(len(df_list))
        # # # Use Pool Multiprocessing
        self.concatenated_dataframe = p.map(self.pool_process_df, df_list)

    def proses_perbandingan_df(self,fileSystem):
        self.set_file_system(fileSystem)
        file_excel = self.file_system.get_lokasi_file_pembanding()
        self.read_from_excel(file_excel)
        self.pool_handler()

    def create_result_file(self):
        nama_asuransi = self.file_system.get_nama_asuransi()
        self.ex.write_to_excel(nama_asuransi,"_result",self.concatenated_dataframe)
        self.file_system.save_perbandingan_model()


    def add_to_provider_list(self):
        self.provider_list.clear()
        if self.df is not None:
            for index, row in self.df.iterrows():
                provider_name = row['Provider Name']
                y_preds = row["Prediction"]
                alamat = row["Alamat"]
                alamat_prediction = row["Alamat Prediction"]
                nil = row["Score"]
                compared = row["Compared"]
                provider_object = ItemPembanding(provider_name, alamat, y_preds, nil, 0, 0, 0)
                provider_object.set_selected(compared)
                provider_object.set_alamat_prediction(alamat_prediction)

                self.provider_list.append(provider_object)

    def get_provider_list(self):
        return self.provider_list

    def get_provider_list_json_response(self):
        ls = []
        for obj in self.provider_list:
            ls.append(obj.__dict__)
        return ls
    # def search(self):
    #     for index_master, row_master in df_master.iterrows():
    #         id = row_master['ProviderId']
    #         provider_name_master = str(row_master['PROVIDER_NAME'])
    #         # provider_name_find = "['" + provider_name_master + "']"
    #
    #         # FIND PREDICTION'S FILE PEMBANDING == NAMA MASTER
    #         # DENGAN ASUMSI PREDICTION DI FILE PEMBANDING SUDAH AKURAT
    #         val = (dfs['Prediction'].str.lower().eq(provider_name_master.lower()))
    #
    #         res = dfs[val]
    #         # print(res.empty)
    #         if not res.empty:
    #             value = res["Prediction"].head(1)
    #             score = res["Score"].head(1)
    #             id_list.append(id)
    #             provider_name_list.append(provider_name_master)
    #             provider_name_predict_list.append(value.values[0])
    #             score_list.append(score.values[0])
    #             prediction_id_object = PredictionId(value.values[0], id)
    #             prediction_list.append(prediction_id_object)


    def create_result_id_file(self):
        dfs = self.get_data_frame()
        df_master = self.read_from_excel("master_provider.xlsx")
        nama_asuransi = self.file_system.get_nama_asuransi()


        id_list = []
        provider_name_list = []
        provider_name_predict_list = []
        score_list = []


        df = pd.DataFrame(
            {'id': id_list, 'Master Name': provider_name_list, 'Prediction': provider_name_predict_list,
             'Score': score_list})

        self.ex.write_to_excel(nama_asuransi,"_result_id.xlsx",dfs)

        return

    def format_maker(self):
        pass
