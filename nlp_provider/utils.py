import pathlib
from multiprocessing import Pool

import pandas as pd
from django.core.files.storage import FileSystemStorage
from tqdm import tqdm

from model.models import Perbandingan


class ItemPembanding:
    def __init__(self,nama_provider,alamat,label_name,proba_score,count_label,ri,rj):
        self.nama_provider = str(nama_provider).strip()
        self.alamat = str(alamat).strip()
        self.label_name = str(label_name).strip()
        self.proba_score = str(proba_score).strip()
        self.count_label_name = str(count_label).strip()
        self.ri = str(ri).strip()
        self.rj = str(rj).strip()

    def get_ri(self):
        if(self.ri == "Y"):
            return "1"
        elif self.ri == "N":
            return "0"

    def get_rj(self):
        if (self.rj == "Y"):
            return "1"
        elif self.rj == "N":
            return "0"


    def set_alamat_prediction(self,alamat_prediction):
        self.alamat_prediction = str(alamat_prediction)

    def get_alamat_prediction(self):
        return self.alamat_prediction

    def set_mapped_times(self,times):
        self.mapped_times = times

    def set_id_master(self,id):
        self.id_master = id


    def get_id_master(self):
        return self.id_master

    def get_mapped_times(self):
        return self.mapped_times

    def set_count_label_name(self,count):
        self.count_label_name = count

    def set_selected(self,selected):
        self.selected = selected

    def set_nama_asuransi(self,nama_asuransi):
        self.nama_asuransi = nama_asuransi


    def get_count_label_name(self):
        return self.count_label_name

    def get_nama_asuransi(self):
        return self.nama_asuransi

    def get_nama_provider(self):
        return self.nama_provider

    def get_alamat(self):
        return self.alamat

    def get_label_name(self):
        return self.label_name

    def get_proba_score(self):
        return self.proba_score

    def get_selected(self):
        return self.selected

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


class PredictionId:
    def __init__(self,prediction,id_master):
        self.prediction = prediction
        self.id_master = id_master

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
        self.set_lokasi_file_pembanding()

    def set_lokasi_file_pembanding(self):
        self.lokasi_file_pembanding = "media/"+self.uploaded_file.name


    def set_extension_file_pembanding(self):
        try:
            self.file_extension = pathlib.Path("media/"+self.uploaded_file).suffix
        except:
            self.file_extension = pathlib.Path("media/" +self.uploaded_file.name).suffix
    def set_nama_asuransi(self,nama_asuransi):
        self.nama_asuransi = nama_asuransi

    def set_uploaded_file(self,w):
        self.uploaded_file = w

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


class FileSystem:
    def __init__(self,filePembandingAsuransi):
        self.file = filePembandingAsuransi
        self.fst = FileSystemStorage()

    def get_path(self):
        return self.fst.path(self.file.get_nama_file_pembanding())


    def save_file(self):
        uploaded_file_name = self.file.get_nama_file_pembanding()
        uploaded_file = self.file.get_uploaded_file()
        if self.allowed_extension() is True:
            self.saved_file = self.fst.save(uploaded_file_name,uploaded_file)
            self.is_the_insurance_ever_compared()
            return True
        else:
            return False

    def get_saved_file(self):
        return "media/"+self.saved_file

    def allowed_extension(self):
        extension = self.file.get_extension_file_pembanding()
        allowed = True if extension == ".xlsx" else False
        return allowed


    def save_perbandingan_to_dashboard(self):
        pass

    def is_the_insurance_ever_compared(self):
        nama_asuransi = self.file.get_nama_asuransi()
        mydata = Perbandingan.objects.filter(nama_asuransi__contains=nama_asuransi).order_by('created_at').values()
        if not mydata:
            perbandingan_model = models.Perbandingan(nama_asuransi=nama_asuransi,
                                                     match_percentage=0,
                                                     status_finish="PROCESSING", file_location=self.get_path())
        else:
            perbandingan_model = Perbandingan.objects.get(pk=mydata[0]["id"])

        self.file.set_perbandingan_model(perbandingan_model)

class Pembersih:
    def __init__(self,df):
        self.df1 = df
        self._rubah_dataframe_astype_str()
        self._hilangkan_tanda_baca()
        self._kecilkan_tulisan()
    def _kecilkan_tulisan(self):
        self.df4 = self.df3.applymap(str.lower)

    def _hilangkan_tanda_baca(self):
        self.df3 = self.df2.replace(to_replace=['\.','\&'],value='',inplace=False,regex=True)

    def _rubah_dataframe_astype_str(self):
        self.df2 = self.df1.astype(str)

    def _return_astype_str(self):
        return self.df

    def _return_df(self):
        return self.df4


class DFHandler:
    def __init__(self,file_system):
        self.file_system = file_system
        self.df = pd.read_excel(self.file_system.get_saved_file())
        self.pembersih = Pembersih(self.df)


    def set_dataframe(self,dataframe):
        self.df = dataframe

    def get_data_frame(self):
        df = self.pembersih._return_df()
        print(df)
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
        # for row in tqdm(df.itertuples(), total=df.shape[0]):
        #     new_string = row.nama
        #     alamat = row.alamat
        #     ri = row.RI
        #     rj = row.RJ
        #
        #     # replace with df_nama_alamat
        #     nama_alamat = row.nama_alamat
        #
        #     provider_name = new_string
        #
        #     # course_title = apotik  klinik kimia farma  cilegon#jl. s.a. tirtayasa no 12
        #
        #     # val = (df_non_duplicate['course_title'].str.lower().str.strip().eq(nama_alamat))
        #     val = (df_non_duplicate['course_title'].eq(nama_alamat))
        #
        #     res = df_non_duplicate[val]
        #
        #     provider_name_list.append(provider_name)
        #     # load the model from disk
        #     sample1 = vectorize_text(nama_alamat, tfidf_vec1)
        #     y_preds = loaded_model1.predict(sample1)
        #     p = loaded_model1.predict_proba(sample1)
        #     ix = p.argmax(1).item()
        #     nil = (f'{p[0, ix]:.2}')
        #
        #     provider_name_predict_list.append(y_preds)
        #     score_list.append(nil)
        #     provider_object = ItemPembanding(provider_name, alamat, y_preds, nil, 0, ri, rj)
        #
        #     if not res.empty:
        #         pred = str(y_preds).replace("[", "").replace("]", "").replace("'", "")
        #         val_master = (df_non_duplicate['subject'].eq(pred))
        #         res_master = df_non_duplicate[val_master]
        #         al = res_master["alamat"].head(1)
        #         try:
        #             alamat_pred = al.values[0]
        #             data_append = {
        #                 "Provider Name": provider_name,
        #                 "Alamat": alamat,
        #                 "Prediction": y_preds,
        #                 "Alamat Prediction": alamat_pred,
        #                 "Score": nil,
        #                 "Compared": 1,
        #                 "Clean": new_string,
        #                 "ri": ri,
        #                 "rj": rj
        #             }
        #             provider_object.set_alamat_prediction(alamat_pred)
        #             df1 = pd.DataFrame(data_append)
        #         except:
        #             print("error")
        #
        #
        #     elif res.empty:
        #
        #         data_append = {
        #             "Provider Name": provider_name,
        #             "Alamat": alamat,
        #             "Prediction": y_preds,
        #             "Alamat Prediction": "-",
        #             "Score": nil,
        #             "Compared": 0,
        #             "Clean": new_string,
        #             "ri": ri,
        #             "rj": rj
        #         }
        #         provider_object.set_alamat_prediction("-")
        #         df1 = pd.DataFrame(data_append)
        #     provider_object_list.append(provider_object)
        #     df_result = df_result.append(df1, ignore_index=True)
        #     # Provider_Perbandingan_data = models.Provider_Perbandingan(nama_asuransi=perbandingan_model.nama_asuransi,
        #     #                                                           perbandingan_id=1,
        #     #                                                           name=provider_name_label, address="-", selected=0)
        #     # Provider_Perbandingan_data.save()

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
        x = p.map(self.pool_process_df, df_list)



