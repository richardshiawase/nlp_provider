import json
import os
import pathlib
import pickle
import re
import shutil
from collections import defaultdict
from functools import reduce
from multiprocessing import Process, Pool

import numpy as np
import requests
from django.core import serializers
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.http import HttpResponse
from django.shortcuts import render
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from requests import Response
from django.http import JsonResponse
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, f1_score, accuracy_score
import django
django.setup()
from model import models
from model.views import create_model
from .utils import ItemPembanding, Prediction, MasterData, PredictionId
from model.models import Provider_Model, Perbandingan, Provider_Perbandingan
from tqdm import tqdm
# from .forms import UploadFileForm
# Create your views here.
df_dataset = pd.read_excel("dataset_excel_copy.xlsx")
new_course_title = df_dataset['course_title'].str.split("#", n=1, expand=True)


def index(request):

    list_pembanding = []

    pembanding_all = models.Perbandingan.objects.all()
    for pembanding in pembanding_all:
        pembanding.file_location = pembanding.file_location.split("media")[1]
        list_pembanding.append(pembanding)

    context = {"list_pembanding":list_pembanding}

    return render(request, 'home.html', context)


def kompilasi(request):


    pembanding = models.Perbandingan.objects.all()
    list_pembandinge = pembanding
    list_pembanding = []
    for pembanding in list_pembandinge:
        pembanding.file_location = pembanding.file_location.split("media")[1]
        list_pembanding.append(pembanding)

    context = {"list_pembanding":list_pembanding}

    return render(request, 'kompilasi.html')


def kompilasi_data(request):
    pembanding_all = models.Perbandingan.objects.all()
    provider_list = []
    for pembanding in pembanding_all:
        pembanding.file_location = pembanding.file_location.split("media")[1]
        dfs = pd.read_excel("media/"+pembanding.file_location_result)
        for index, row in dfs.iterrows():
            alamat = row['Alamat']
            alamat_prediksi = row['Alamat Prediction']
            item_obj = ItemPembanding(row['Provider Name'], row['Alamat'], row["Prediction"], row["Score"], 0)
            item_obj.set_nama_asuransi(pembanding.nama_asuransi)
            item_obj.set_selected(str(row['Compared']))
            item_obj.set_alamat_prediction(alamat_prediksi)

            provider_list.append(item_obj.__dict__)

    return JsonResponse(provider_list, safe=False)





def newe(request):
    data = list(models.Perbandingan.objects.values())
    if request.method == "GET":
        return JsonResponse(data, safe=False)

    return JsonResponse({'message':'error'})




def perbandingan_rev(request):
    global provider_liste
    global file_location
    provider_liste = []
    response = requests.get('https://asateknologi.id/api/insuranceall')
    response = response.json()
    dfs = None
    prediction_dict = {}
    prediction_dict = defaultdict(lambda:0,prediction_dict)
    if request.method == "POST":
        file_location = "media"+request.POST["file_location"]


    try:
        dfs = pd.read_excel(file_location)
    except:
        print("dataframe not founds")
    provider_list = []
    if dfs is not None:
        for index, row in dfs.iterrows():
            provider_name = row['Provider Name']
            y_preds = row["Prediction"]
            alamat = row['Alamat']
            alamat_prediksi = row['Alamat Prediction']
            nil = row["Score"]
            compared = row["Compared"]
            # city = row["City"]
            # print(city)
            prediction_dict[y_preds] += 1

            provider_object = ItemPembanding(provider_name, alamat, y_preds, nil, 0)
            provider_object.set_selected(compared)
            provider_object.set_alamat_prediction(alamat_prediksi)
            provider_list.append(provider_object.__dict__)


    # MAP THE COUNT !
    for provider_dict in provider_list:
        for key, values in prediction_dict.items():
            if(key == provider_dict["label_name"]):
                provider_dict['count_label_name'] = values


    return JsonResponse(provider_list, safe=False)



def perbandingan(request):
    global provider_liste
    global file_location
    # file_location = "-"
    provider_liste = []
    response = requests.get('https://asateknologi.id/api/insuranceall')
    response = response.json()
    dfs = None

    if request.method == "POST":
        file_location = "media"+request.POST["file_location"]
        print(file_location)
    # elif request.method == "GET":
        # file_location="media/demo.xlsx"
    # else:
    #     file_location = "-"
    try:
        dfs = pd.read_excel(file_location)
    except:
        print("dataframe not founde")
    provider_list = []
    if dfs is not None:
        for index, row in dfs.iterrows():
            provider_name = row['Provider Name']
            y_preds = row["Prediction"]
            alamat = row["Alamat"]
            alamat_prediction = row["Alamat Prediction"]
            nil = row["Score"]
            compared = row["Compared"]
            provider_object = ItemPembanding(provider_name, alamat, y_preds, nil, 0)
            provider_object.set_selected(compared)
            provider_object.set_alamat_prediction(alamat_prediction)

            provider_list.append(provider_object)


    context = {"list_insurance":response.get("val"),"list":provider_list,"link_result":file_location}


    return render(request, 'matching/perbandingan.html',context=context)


def tampungan(request):
    context = {"provider_list":[]}
    return render(request, 'matching/perbandingan_basket.html', context=context)


def tampungan_rev(request):
    global provider_liste
    global file_location
    provider_liste = []

    dfs = None

    try:
        dfs = pd.read_excel("basket_provider.xlsx")
    except:
        print("dataframe not founde")

    filename = 'tfidf_vec.pickle'
    tfidf_vec = pickle.load(open(filename, 'rb'))
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    provider_name_list = []
    provider_name_predict_list = []
    score_list = []
    # df_dataset = pd.read_excel("dataset_excel_copy.xlsx")

    provider_list = []
    if dfs is not None:
        for index, row in dfs.iterrows():
            new = df_dataset['course_title'].str.split("#", n=1, expand=True)
            df_dataset["course_titles"] = new[0]


            provider_name_label = str(row['course_title']).strip().lower()
            alamat = str(row['alamat']).strip().lower()
            concat = provider_name_label+"#"+alamat
            concat = concat.replace('&','').replace('.','')
            sample1 = vectorize_text(concat, tfidf_vec)
            y_preds = loaded_model.predict(sample1)
            p = loaded_model.predict_proba(sample1)
            ix = p.argmax(1).item()
            nil = (f'{p[0, ix]:.2}')
            # if(float(nil.strip("%")) < 1.0):
            # y_preds = "-"
            provider_name_list.append(provider_name_label)
            provider_name_predict_list.append(y_preds)
            score_list.append(nil)

            val = (df_dataset['course_titles'].str.lower().eq(provider_name_label))
            res = df_dataset[val]
            provider_object = ItemPembanding(provider_name_label, alamat, y_preds, nil, 0)

            if not res.empty:
                pred = str(y_preds).replace("[", "").replace("]", "").replace("'", "")
                val_master = (df_dataset['subject'].eq(pred))
                res_master = df_dataset[val_master]

                al = res_master["alamat"].head(1)

                alamat_pred = al.values[0]
            elif res.empty:
                alamat_pred = "-"


            provider_object.set_alamat_prediction(alamat_pred)

            provider_list.append(provider_object.__dict__)

    return JsonResponse(provider_list,safe=False)


def hapus_tampungan(request):
    dfs = pd.read_excel("basket_provider.xlsx")

    if request.method == "POST":
        nama = request.POST['nama_provider']
        # nama = json.load(request).get('dats')
        print(nama.upper())
        delete_row = dfs[dfs["course_title"] == nama.upper()].index
        dfs = dfs.drop(delete_row)
        # print(dfs)
        dfs.to_excel('basket_provider.xlsx',index=False)


    return HttpResponse("OK")

def upload_master(request):
    global provider_liste
    global file_location
    provider_liste = []
    response = requests.get('https://asateknologi.id/api/insuranceall')
    response = response.json()
    dfs = None

    if request.method == "POST":
        file_location = "media"+request.POST["file_location"]

    # elif request.method == "GET":
        # file_location="media/demo.xlsx"

    try:
        dfs = pd.read_excel(file_location)
    except:
        print("dataframe not found")
    provider_list = []
    if dfs is not None:
        for index, row in dfs.iterrows():
            provider_name = row['Provider Name']
            alamat = row['Alamat']
            alamat_prediction = row['Alamat Prediction']
            y_preds = row["Prediction"]
            nil = row["Score"]
            provider_object = ItemPembanding(provider_name, alamat, y_preds, nil, 0)
            provider_object.set_alamat_prediction(alamat_prediction)
            provider_list.append(provider_object)
    page = request.GET.get('page', 1)
    paginator = Paginator(provider_list, 10)
    try:
        users = paginator.page(page)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)

    context = {"list_insurance":response.get("val"),"list":provider_list}


    return render(request, 'master/bulk_upload.html',context=context)


def list_master(request):


    return render(request, 'master/index.html')


def list_master_varian(request):
    return render(request, 'master/index_master_varian.html')

def list_master_sinkron(request):

    return render(request, 'master/sinkron.html')

def list_master_process(request):
    master_data_list = []
    dfs  = None
    try:
        dfs = pd.read_excel("master_provider.xlsx")
    except:
        print("dataframe not found")

    for index, row in dfs.iterrows():
        id = row['ProviderId']
        stateId = row['stateId']
        cityId = row['cityId']
        provider_name_master = str(row['PROVIDER_NAME'])
        address = row['ADDRESS']
        category_1 = row['Category_1']
        category_2 = row['Category_2']
        telephone = row['TEL_NO']
        master_data = MasterData(id, provider_name_master, address, category_1, category_2, telephone, stateId,
                                 cityId)
        master_data_list.append(master_data.__dict__)
    return JsonResponse(master_data_list, safe=False)



def sinkron_master_process(request):

    response = requests.get('https://asateknologi.id/api/daftar-rs-1234')
    provider_list = response.json().get("val")
    master_data_list = []
    df = pd.DataFrame()

    for prov in provider_list:
        id = prov["id"]
        stateId = prov["stateId"]
        cityId = prov["CityId"]
        category_1 = str(prov["Category_1"])
        category_2 = prov["Category_2"]
        telephone = prov["TEL_NO"]
        provider_name_master = prov["PROVIDER_NAME"]
        address = prov["ADDRESS"]
        category = prov["Category_1"]
        master_data = MasterData(id,provider_name_master,address,category_1,category_2,telephone,stateId,cityId)
        master_data_list.append(master_data.__dict__)
        df = df.append(pd.Series(
            {'ProviderId':id,'stateId':stateId,'cityId':cityId,'Category_1':category_1,'Category_2':category_2,'PROVIDER_NAME': provider_name_master, 'ADDRESS':address, 'TEL_NO': telephone},
            name=3))



    df.to_excel("master_provider.xlsx", index=False)


    return JsonResponse(master_data_list, safe=False)

def download_master(request):

    file_path = os.getcwd()+"\\master_provider.xlsx"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(),
                                    content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            response['Content-Disposition'] = 'attachment; filename=master_provider.xlsx'
            return response
    else:
        raise None
    return response


def download_master_varian(request):

    file_path = os.getcwd()+"\\master_varian_1.xlsx"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(),
                                    content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            response['Content-Disposition'] = 'attachment; filename=master_varian.xlsx'
            return response
    else:
        raise None
    return response

def sinkron_dataset_process(request):
    dff = pd.DataFrame()

    find = False
    master_data_list = []
    dfs = None
    dfs_varian = None
    try:
        dfs = pd.read_excel("master_provider.xlsx")
        df = pd.read_excel("dataset_excel_copy.xlsx")
        dfs_varian = pd.read_excel("dataset_excel_copy.xlsx").groupby('subject')
    except:
        print("dataframe not found")
    for index, row in dfs.iterrows():
        id = row['ProviderId']
        stateId = row['stateId']
        cityId = row['cityId']
        category_1 = row['Category_1']
        category_2 = row['Category_2']
        provider_name_master = row['PROVIDER_NAME']
        address = row['ADDRESS']
        tel_no = row['TEL_NO']
        master_data = MasterData(id, provider_name_master, address, category_1, category_2, tel_no, stateId, cityId)
        varian_list = []

        try:
            dfe = dfs_varian.get_group(provider_name_master)
            for index_varian, row_varian in dfe.iterrows():
                varian_list.append(row_varian['course_title'])
                pass

        except:
            row = pd.Series({'course_title': master_data.nama_provider + "#" + master_data.alamat, 'alamat':master_data.alamat,'subject': master_data.nama_provider},
                            name=3)
            df = df.append(row)
            df.reset_index(drop=True, inplace=True)

            continue


    df.to_excel("dataset_excel_copy.xlsx",index=False)
    return HttpResponse("Tes")



def master_varian_process(request):
    dff = pd.DataFrame()

    find = False
    master_data_list = []
    dfs  = None
    dfs_varian = None
    try:
        dfs = pd.read_excel("master_provider.xlsx")
        dfs_varian = pd.read_excel("dataset_excel_copy.xlsx").groupby('subject')
    except:
        print("dataframe not found")

    for index, row in dfs.iterrows():
        id = row['ProviderId']
        stateId = row['stateId']
        cityId = row['cityId']
        category_1 = row['Category_1']
        category_2 = row['Category_2']
        provider_name_master = row['PROVIDER_NAME']
        address = row['ADDRESS']
        tel_no = row['TEL_NO']
        master_data = MasterData(id, provider_name_master, address, category_1, category_2, tel_no, stateId, cityId)
        varian_list = []


        try:
            dfe = dfs_varian.get_group(provider_name_master)
            for index_varian, row_varian in dfe.iterrows():
                varian_list.append(row_varian['course_title'])
                pass

        except:
            continue


        master_data.set_varian(varian_list)

        dff = dff.append(pd.Series(
            {'ProviderId': id,'ProviderType':"Master", 'stateId': stateId, 'cityId': cityId, 'Category_1': category_1, 'Category_2': category_2,
             'PROVIDER_NAME': provider_name_master, 'ADDRESS': address, 'TEL_NO': tel_no},
            name=3))

        for varian in master_data.get_varian():
            dff = dff.append(pd.Series(
                {'ProviderId': id, 'ProviderType': "Varian", 'stateId': stateId, 'cityId': cityId,
                 'Category_1': category_1, 'Category_2': category_2,
                 'PROVIDER_NAME': varian, 'ADDRESS': "-", 'TEL_NO': "-"},
                name=3))

        master_data_list.append(master_data.__dict__)

    #
    dff.to_excel("master_varian_1.xlsx", index=False)

    return JsonResponse(master_data_list, safe=False)

def master_varian_list_read(request):
    dff = pd.DataFrame()

    find = False
    master_data_list = []
    dfs  = None
    try:
        dfs = pd.read_excel("master_varian_1.xlsx")
    except:
        print("dataframe not found")

    for index, row in dfs.iterrows():
        id = row['ProviderId']
        stateId = row['stateId']
        cityId = row['cityId']
        category_1 = row['Category_1']
        category_2 = row['Category_2']
        provider_name_master = row['PROVIDER_NAME']
        address = row['ADDRESS']
        tel_no = row['TEL_NO']
        master_data = MasterData(id, provider_name_master, address, category_1, category_2, tel_no, stateId, cityId)
        varian_list = []

        try:
            dfe = dfs_varian.get_group(provider_name_master)
            for index_varian, row_varian in dfe.iterrows():
                varian_list.append(row_varian['course_title'])
                pass

        except:
            continue


        master_data.set_varian(varian_list)

        dff = dff.append(pd.Series(
            {'ProviderId': id,'ProviderType':"Master", 'stateId': stateId, 'cityId': cityId, 'Category_1': category_1, 'Category_2': category_2,
             'PROVIDER_NAME': provider_name_master, 'ADDRESS': address, 'TEL_NO': tel_no},
            name=3))

        for varian in master_data.get_varian():
            dff = dff.append(pd.Series(
                {'ProviderId': id, 'ProviderType': "Varian", 'stateId': stateId, 'cityId': cityId,
                 'Category_1': category_1, 'Category_2': category_2,
                 'PROVIDER_NAME': varian, 'ADDRESS': "-", 'TEL_NO': "-"},
                name=3))

        master_data_list.append(master_data.__dict__)

    #
    dff.to_excel("master_varian_1.xlsx", index=False)

    return JsonResponse(master_data_list, safe=False)

def temporer_store(request):
    global link_result
    if request.method == "POST":
        global name

        post_ide = request.POST["post_idew"]
        alamat = request.POST["alamat"]
        name = post_ide + "#"+alamat
        link_result = request.POST["link_result"]
        context = {"provider_name": post_ide,"link_result":link_result}


        if name in provider_liste:
            provider_liste.remove(name)
        else:
            provider_liste.append(name)
    else :
        context = {"provider_name":provider_liste,"link_result":link_result}

    # return HttpResponse(context)
    return render(request,'matching/temporer.html',context=context)


def add_master_store(request):
    if request.method == "POST":
        df = pd.read_excel("Master_Add.xlsx")
        post_ide = request.POST["post_idew"]
        nama_provider = post_ide.split("#")[0]
        alamat = post_ide.split("#")[1]
        link_result = request.POST["link_result"]
        val = (df['provider_name'].str.lower().eq(nama_provider.lower()))
        res = df[val]

        if res.empty:
            row = pd.Series({'provider_name': nama_provider, 'alamat': alamat})
            df = df.append(row, ignore_index=True)
            df.to_excel("Master_Add.xlsx",index=False)
            dfs = pd.read_excel(link_result)
            val = (dfs['Provider Name'].eq(nama_provider.upper()))
            rese = dfs[val]
            if not rese.empty:
                print(rese.index.item())
                deo = dfs.drop(rese.index.item())
                deo.to_excel(link_result, sheet_name='Sheet1', index=False)
                dat = Perbandingan.objects.filter(file_location_result__contains=link_result.split("/")[1]).values()
                print(dat[0]["file_location"],os.getcwd())
                dw = pd.read_excel(dat[0]["file_location"])
                val = (dw['Nama Provider'].eq(nama_provider.upper()))
                reseq = dw[val]
                if not reseq.empty:
                    deoq = dw.drop(reseq.index.item())
                    deoq.to_excel(dat[0]["file_location"], sheet_name='Sheet1', index=False)




    else:
        return HttpResponse("OK")



    # return HttpResponse(context)
    return HttpResponse("OK")

def update_temporer_store(request):
    global name
    global link_result
    if request.method == "POST":
        post_ide = request.POST["post_idew"]
        link_result = request.POST["link_result"]
        name = post_ide
        context = {"provider_name": post_ide,"link_result":link_result}
    else :
        if name in provider_liste:
            provider_liste.remove(name)
        context = {"provider_name":provider_liste,"link_result":link_result}
    return render(request,'matching/temporer.html',context=context)


def add_to_dataset(request):
    if request.method == "POST":
        # OPEN DATASET FILE
        df = pd.read_excel("dataset_excel_copy.xlsx")
        df_basket = pd.read_excel("basket_provider.xlsx")

        # SEARCH PROVIDER IN DATASET
        for label_name,provider_name in list(zip(request.POST.getlist('nama_label'),request.POST.getlist('nama_provider'))):
            label_name = label_name.split("#")[0]
            alamat = provider_name.split("#")[1]

            provider_name = provider_name.split("#")[0]
            for x in range(200):
                try:
                    row = pd.Series({'course_title': provider_name+"#"+alamat, 'subject': label_name}, name=3)
                    df = df.append(row,ignore_index=True)
                except:
                    break
            try:
                rowe = pd.Series({'course_title': provider_name,'alamat':alamat}, name=3)
                df_basket = df_basket.append(rowe, ignore_index=True)
            except:
                break

        # df = df.reset_index(drop=True)
        df_basket.to_excel("basket_provider.xlsx",index=False)
        df.to_excel("dataset_excel_copy.xlsx",index=False)
        # create_model(df)

        pembanding = models.Perbandingan.objects.all()
        list_pembandinge = pembanding
        list_pembanding = []
        for pembanding in list_pembandinge:
            pembanding.file_location = pembanding.file_location.split("media")[1]
            list_pembanding.append(pembanding)

        context = {"list_pembanding": list_pembanding}

        return render(request, 'home.html', context)

    return HttpResponse("Marco Polo")

def process_temporer_store(request):
    global link_result
    if request.method == "POST":
        link_result  = request.POST["link_result"]
    dfs = pd.read_excel("dataset_excel_copy.xlsx")
    # dfs = dfs.sort_values(by=['subject'], ascending = True)
    # dfa = dfs.drop_duplicates(subset='subject')
    dfz = dfs.dropna(subset="alamat")
    dfa = dfz.drop_duplicates(subset='subject')
    label_list = []
    for index, row in dfa.iterrows():
        provider_name = row['course_title']
        alamat = str(row['alamat'])
        label = row["subject"]
        if label+"#"+alamat not in label_list:
            label_list.append(label+"#"+alamat)
    print(link_result)
    context = {"label_list":label_list,"list":provider_liste,"link_result":link_result}
    # return HttpResponse("Process Temporer")
    return render(request,'matching/proses_temporer.html',context=context)


def get_label(request):
    dfs = pd.read_excel("dataset_excel_copy.xlsx")
    dfs = dfs.sort_values(by=['subject'], ascending=True)
    dfz = dfs.dropna(subset="alamat")
    dfa = dfz.drop_duplicates(subset='subject')
    label_list = []
    print(dfa.size,dfs.size)
    for index, row in dfa.iterrows():
        provider_name = row['course_title']
        alamat = str(row['alamat'])
        label = row["subject"]
        if label + "#" + alamat not in label_list:
            label_list.append(label + "#" + alamat)

    context = {"label_list": label_list}
    return JsonResponse(context, safe=False)


def check_header(df):
    header_list = ['Provinsi','Kota','Nama Provider','Alamat']
    df_header_list = list(df.columns.values)
    if df_header_list == header_list:
        return True
    return False

def vectorize_text(text,tfidf_vec):
    # text = "Klinik Ananda"
    my_vec = tfidf_vec.transform([text])
    return my_vec.toarray()

def pool_process_df(df):
    # for df in df_list:
    # dataframe Name and Age columns
    pd.options.display.max_colwidth = None
    provider_name_list = []
    provider_name_predict_list = []
    score_list = []
    provider_object_list = []


    df_result = pd.DataFrame()
    # df_dataset = pd.read_excel("dataset_excel_copy.xlsx")
    filename = 'tfidf_vec.pickle'
    tfidf_vec = pickle.load(open(filename, 'rb'))
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        new_string = (row._3).strip().lower()
        alamat = str(row.Alamat).strip().lower()
        value = new_string + "#" + alamat
        new_string = value.replace('&', '')
        new_string = new_string.replace('.', '')
        provider_name = new_string.replace('-', '')
        provider_name_label = row._3
        df_dataset["course_titles"] = new_course_title[0]
        val = (df_dataset['course_titles'].str.lower().eq(provider_name_label.split("#")[0].lower()))
        res = df_dataset[val]


        provider_name_list.append(provider_name_label)
        # load the model from disk
        sample1 = vectorize_text(new_string, tfidf_vec)
        y_preds = loaded_model.predict(sample1)
        p = loaded_model.predict_proba(sample1)
        ix = p.argmax(1).item()
        nil = (f'{p[0, ix]:.2}')


        provider_name_predict_list.append(y_preds)
        score_list.append(nil)
        provider_object = ItemPembanding(provider_name, alamat, y_preds, nil, 0)

        if not res.empty:
            pred = str(y_preds).replace("[", "").replace("]", "").replace("'", "")
            val_master = (df_dataset['subject'].eq(pred))
            res_master = df_dataset[val_master]

            al = res_master["alamat"].head(1)

            alamat_pred = al.values[0]
            data_append = {
                "Provider Name": provider_name_label,
                "Alamat": alamat,
                "Prediction": y_preds,
                "Alamat Prediction": alamat_pred,
                "Score": nil,
                "Compared": 1,
                "Clean": new_string
            }
            provider_object.set_alamat_prediction(alamat_pred)
            df1 = pd.DataFrame(data_append)

        elif res.empty:

            data_append = {
                "Provider Name": provider_name_label,
                "Alamat": alamat,
                "Prediction": y_preds,
                "Alamat Prediction": "-",
                "Score": nil,
                "Compared": 0,
                "Clean": new_string
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

def pool_handler(df,perbandingan_model):

    # # # Split dataframe to many
    df_list = cacah_dataframe(df)

    # # # Using multiprocess with pool as many as dataframe list
    p = Pool(len(df_list))

    # # # Use Pool Multiprocessing
    x = p.map(pool_process_df,df_list)

    # # # Declare write
    writer = pd.ExcelWriter('media/' + perbandingan_model.nama_asuransi + "_result" + ".xlsx",
                            engine='xlsxwriter')

    # # # Save Perbandingan Model
    perbandingan_model.file_location_result = "/" + perbandingan_model.nama_asuransi + "_result.xlsx"
    perbandingan_model.save()

    # # # Concat list of dataframe
    full_dfw = pd.concat(list(x),ignore_index=True)

    # # # Convert the dataframe to an XlsxWriter Excel object.
    full_dfw.to_excel(writer, sheet_name='Sheet1', index=False)

    # # # Close the Pandas Excel writer and output the Excel file.
    writer.close()


def cacah_dataframe(df):
    split_row_each = 600
    start_index = 0
    iteration_count = int(df.shape[0]/split_row_each)
    sisa = df.shape[0]%split_row_each
    sisa_row = iteration_count*split_row_each+sisa
    df_list = []
    for x in range(iteration_count):
        end_index = start_index + split_row_each
        df_new= df.iloc[start_index:end_index]
        start_index = end_index
        # df_list.append([df_new,lr])
        df_list.append(df_new)
    aw = lambda x,y : y if x > 0 else 0
    df_last = df.iloc[start_index:aw(sisa, sisa_row)]
    df_list.append(df_last)

    return df_list




def is_file_with_this_insurance_exists(nama_asuransi):
    mydata = Perbandingan.objects.filter(nama_asuransi__contains=nama_asuransi).order_by('created_at').values()
    return mydata


def create_result_file(dfs,prediction_list):
    df_master = pd.read_excel("master_provider.xlsx")
    id_list = []
    provider_name_list = []
    provider_name_predict_list = []
    score_list = []

    writerez = pd.ExcelWriter('media/' + perbandingan_model.nama_asuransi + "_result_id.xlsx", engine='xlsxwriter')
    for index_master, row_master in df_master.iterrows():
        id = row_master['ProviderId']
        provider_name_master = str(row_master['PROVIDER_NAME'])
        # provider_name_find = "['" + provider_name_master + "']"

        # FIND PREDICTION'S FILE PEMBANDING == NAMA MASTER
        # DENGAN ASUMSI PREDICTION DI FILE PEMBANDING SUDAH AKURAT
        val = (dfs['Prediction'].str.lower().eq(provider_name_master.lower()))


        res = dfs[val]
        # print(res.empty)
        if not res.empty:

            value = res["Prediction"].head(1)
            score = res["Score"].head(1)
            id_list.append(id)
            provider_name_list.append(provider_name_master)
            provider_name_predict_list.append(value.values[0])
            score_list.append(score.values[0])
            prediction_id_object = PredictionId(value.values[0], id)
            prediction_list.append(prediction_id_object)
    df = pd.DataFrame(
        {'id': id_list, 'Master Name': provider_name_list, 'Prediction': provider_name_predict_list,
         'Score': score_list})
    # # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writerez, sheet_name='Sheet1', index=False)
    # # Close the Pandas Excel writer and output the Excel file.
    writerez.close()
    return

def create_result_file_final(dfs,prediction_list):
    writere = pd.ExcelWriter('media/' + perbandingan_model.nama_asuransi + "_result_final.xlsx", engine='xlsxwriter')

    provider_list = []
    provider_name_list_final = []
    id_list_final = []
    provider_name_predict_list_final = []
    score_list_final = []
    for index, row in dfs.iterrows():
        provider_name = row['Provider Name']
        y_preds = row["Prediction"]
        nil = row["Score"]
        alamat = row["Alamat"]
        alamat_pred = row["Alamat Prediction"]
        provider_object = ItemPembanding(provider_name, alamat, y_preds, nil, 0)
        provider_object.set_id_master("-")
        provider_object.set_alamat_prediction(alamat_pred)

        for preds in prediction_list:
            # COMPARE FILE PEMBANDING  dengan PREDICTION VALUE DENGAN ASUMSI PEMBANDING UDAH BENER
            if preds.prediction == provider_object.get_label_name():
                provider_object.set_id_master(preds.id_master)

        provider_name_list_final.append(provider_object.get_nama_provider())
        provider_name_predict_list_final.append(provider_object.get_label_name())
        score_list_final.append(provider_object.get_proba_score())
        id_list_final.append(provider_object.get_id_master())

        provider_list.append(provider_object)

    df = pd.DataFrame(
        {'id_master': id_list_final, 'Provider Name': provider_name_list_final,
         'Prediction': provider_name_predict_list_final,
         'Score': score_list_final,'ri':'0','rj':'0'})
    # # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writere, sheet_name='Sheet1', index=False)
    # # Close the Pandas Excel writer and output the Excel file.
    writere.close()
    return provider_list


def update_perbandingan_excel():
    pass

def perbandingan_result(request):
    global uploaded_file
    global contexte
    if request.method == 'POST':
        uploaded_file = None
        file_extension = None
        filename = None
        fs = FileSystemStorage()
        if not bool(request.FILES.get('perbandinganModel',False)) :
            uploaded_file = request.POST['perbandinganModelFile']
            file_extension = pathlib.Path("media/"+uploaded_file).suffix
            file_content = pathlib.Path("media/"+uploaded_file)
            filename = file_content

        else:
            uploaded_file = request.FILES['perbandinganModel']
            file_extension = pathlib.Path("media/" + uploaded_file.name).suffix
            filename = fs.save(uploaded_file.name, uploaded_file)
        # print(uploaded_file)
        menu_insurance = request.POST['insurance_option']

        if file_extension != ".xlsx":
            return HttpResponse("Extension / Format tidak diizinkan")
        uploaded_file_path = fs.path(filename)

        # shutil.copyfile(uploaded_file.name, os.getcwd())

        insurance_data = is_file_with_this_insurance_exists(menu_insurance)
        global perbandingan_model

        if not insurance_data:
            perbandingan_model = models.Perbandingan(nama_asuransi=menu_insurance, match_percentage=0,
                                                     status_finish="PROCESSING", file_location=uploaded_file_path)
        else:
            perbandingan_model = Perbandingan.objects.get(pk=insurance_data[0]["id"])

        df = pd.read_excel(uploaded_file)
        pool_handler(df,perbandingan_model)


        dfs = pd.read_excel("media/"+perbandingan_model.file_location_result)
        prediction_list = []

        create_result_file(dfs,prediction_list)
        provider_list = create_result_file_final(dfs,prediction_list)
        print(perbandingan_model.file_location_result)
        contexte = {"list":provider_list,"link_result":"media/"+perbandingan_model.file_location_result}
        return render(request, 'matching/perbandingan.html', context=contexte)

    return render(request, 'matching/perbandingan.html',context=contexte)
