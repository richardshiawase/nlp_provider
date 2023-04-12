import json
import os
import pathlib
import pickle
import re
import shutil
from collections import defaultdict
from functools import reduce
from multiprocessing import Process, Pool
from time import sleep

import numpy as np
import requests
from django.core import serializers
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.forms import model_to_dict
from django.http import HttpResponse
from django.shortcuts import render
import warnings

from classM.DFHandler import DFHandler
from classM.Dataset import Dataset
from model.models import ItemProvider, List_Processed_Provider
from classM.MasterData import MasterData
from classM.Pembersih import Pembersih
from classM.PerbandinganResult import PerbandinganResult
from classM.PredictionId import PredictionId
from classM.ColumnToRead import ColumnToRead

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import pandas as pde
from requests import Response
from django.http import JsonResponse
import django

django.setup()
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, f1_score, accuracy_score

from model import models
from model.views import create_model

from model.models import Provider_Model, Provider
from tqdm import tqdm
from django.core.cache import cache

# from .forms import UploadFileForm
# Create your views here.

dataset = Dataset(pd)
print(dataset.get_bulk_dataset()['course_title'])
list_provider_model_object = List_Processed_Provider()
provider_dict_item = {}



# new_course_title = df_dataset['course_title'].str.lower().str.split("#", n=1, expand=True)
# df_dataset["course_titles"] = new_course_title[0]
# p = Pembersih((df_dataset.drop_duplicates(['course_title'], keep='first')))

df_non_duplicate = dataset.get_dataframe_after_cleaned_no_duplicate()
df_handler = DFHandler()

filename = 'tfidf_vec.pickle'
tfidf_vec1 = pickle.load(open(filename, 'rb'))
filename = 'finalized_model.sav'
loaded_model1 = pickle.load(open(filename, 'rb'))


def index(request):
    context = {"list_pembanding": []}
    return render(request, 'home.html', context)


def kompilasi(request):
    pembanding = models.Provider.objects.all()
    list_pembandinge = pembanding
    list_pembanding = []
    for pembanding in list_pembandinge:
        pembanding.file_location = pembanding.file_location.split("media")[1]
        list_pembanding.append(pembanding)

    return render(request, 'kompilasi.html')


def kompilasi_data(request):
    pembanding_all = models.Provider.objects.all()
    provider_list = []
    for pembanding in pembanding_all:
        pembanding.file_location = pembanding.file_location.split("media")[1]
        dfs = pd.read_excel("media/" + pembanding.file_location_result)
        for index, row in dfs.iterrows():
            alamat = row['Alamat']
            alamat_prediksi = row['Alamat Prediction']
            ri = row['ri']
            rj = row['rj']
            item_obj = ItemProvider(row['Provider Name'], row['Alamat'], row["Prediction"], row["Score"], 0, ri, rj)
            item_obj.set_nama_asuransi(pembanding.nama_asuransi)
            item_obj.set_selected(str(row['Compared']))
            item_obj.set_alamat_prediction(alamat_prediksi)

            provider_list.append(item_obj.__dict__)

    return JsonResponse(provider_list, safe=False)


def newe(request):
    list_providere = []

    data_list = models.Provider.objects.raw(
        "select * from model_provider where created_at in (select max(created_at) from model_provider group by nama_asuransi)")
    provider_list = []
    for data in data_list:
        pk = data.pk
        provider = Provider()
        provider.set_nama_asuransi_model(data.nama_asuransi)
        provider.set_file_location(data.file_location)

        provider.set_id(pk)
        provider.status_finish = data.status_finish
        provider.match_percentage = data.match_percentage
        provider.file_location_result = data.file_location_result

        list_item_provider = []
        dt = models.Provider.objects.raw("select * from model_itemprovider where id_model = %s", [pk])

        for item in dt:
            item._state.adding = False
            item_provider = ItemProvider()
            item_provider.set_id(item.pk)
            item_provider.set_provider_name(item.nama_provider)
            item_provider.set_alamat_prediction(item.alamat_prediction)
            item_provider.set_alamat(item.alamat)
            item_provider.set_proba_score(item.proba_score)
            item_provider.set_label_name(item.label_name)
            item_provider.set_ri(item.ri)
            item_provider.set_rj(item.rj)
            item_provider.set_id_asuransi(item.id_asuransi)
            item_provider.set_selected("-")
            list_item_provider.append(item_provider)

        provider.set_list_item_provider(list_item_provider)
        provider_list.append(provider)

    list_provider_model_object.set_provider_list(provider_list)

    for item in list_provider_model_object.get_provider_list():
        data = model_to_dict(item)
        list_providere.append(data)

    if request.method == "GET":
        return JsonResponse(list_providere, safe=False)

    return JsonResponse({'message': 'error'})


def perbandingan_rev(request):
    id_provider = request.session.get('id_provider')
    provider = list_provider_model_object.get_a_provider_from_id(id_provider)

    return JsonResponse(provider.get_list_item_provider_json(), safe=False)


def perbandingan(request):
    response = requests.get('https://asateknologi.id/api/insuranceall')
    response = response.json()

    if request.method == "POST":
        id_provider = request.POST["id_provider"]
        request.session['id_provider'] = id_provider
        # loop_delete(file_location)

    context = {"list_insurance": response.get("val"), "list": [], "link_result": "-"}
    return render(request, 'matching/perbandingan.html', context=context)


def tampungan(request):
    link_result = file_location
    if link_result is None:
        link_result = "-"

    context = {"provider_list": [], "link_result": link_result}
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

    provider_name_list = []
    provider_name_predict_list = []
    score_list = []
    # df_dataset = pd.read_excel("dataset_excel_copy.xlsx")

    provider_list = []
    if dfs is not None:
        for index, row in tqdm(dfs.iterrows(), total=dfs.shape[0]):

            provider_name_label = str(row['course_title']).strip().lower()
            alamat = str(row['alamat']).strip().lower()
            concat = provider_name_label + "#" + alamat
            concat = concat.replace('&', '').replace('.', '')
            sample1 = vectorize_text(concat, tfidf_vec1)
            y_preds = loaded_model1.predict(sample1)
            p = loaded_model1.predict_proba(sample1)
            ix = p.argmax(1).item()
            nil = (f'{p[0, ix]:.2}')
            # if(float(nil.strip("%")) < 1.0):
            # y_preds = "-"
            provider_name_list.append(provider_name_label)
            provider_name_predict_list.append(y_preds)
            score_list.append(nil)

            val = (df_non_duplicate['course_titles'].eq(provider_name_label))
            res = df_non_duplicate[val]
            provider_object = ItemProvider(provider_name_label, alamat, y_preds, nil, 0, 0, 0)

            if not res.empty:
                pred = str(y_preds).replace("[", "").replace("]", "").replace("'", "")
                val_master = (df_non_duplicate['subject'].eq(pred))
                res_master = df_non_duplicate[val_master]

                al = res_master["alamat"].head(1)
                try:
                    alamat_pred = al.values[0]
                except:
                    print("error")
            elif res.empty:
                alamat_pred = "-"

            provider_object.set_alamat_prediction(alamat_pred)

            provider_list.append(provider_object.__dict__)

    return JsonResponse(provider_list, safe=False)


def hapus_tampungan(request):
    dfs = pd.read_excel("basket_provider.xlsx")

    if request.method == "POST":
        nama = request.POST['nama_provider']
        # nama = json.load(request).get('dats')
        print(nama.upper())
        delete_row = dfs[dfs["course_title"] == nama.upper()].index
        dfs = dfs.drop(delete_row)
        # print(dfs)
        dfs.to_excel('basket_provider.xlsx', index=False)

    return HttpResponse("OK")


def upload_master(request):
    global provider_liste
    global file_location
    provider_liste = []
    response = requests.get('https://asateknologi.id/api/insuranceall')
    response = response.json()
    dfs = None

    if request.method == "POST":
        file_location = "media" + request.POST["file_location"]

    # elif request.method == "GET":
    # file_location="media/demo.xlsx"

    try:
        dfs = pd.read_excel(file_location)
    except:
        print("dataframe not found")
    provider_list = []
    # # # MASUKKAN DF KE LIST
    if dfs is not None:
        for index, row in dfs.iterrows():
            provider_name = row['Provider Name']
            alamat = row['Alamat']
            alamat_prediction = row['Alamat Prediction']
            y_preds = row["Prediction"]
            nil = row["Score"]
            ri = row["RI"]
            rj = row["RJ"]
            provider_object = ItemProvider(provider_name, alamat, y_preds, nil, 0, ri, rj)
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

    context = {"list_insurance": response.get("val"), "list": provider_list}

    return render(request, 'master/bulk_upload.html', context=context)


def list_master(request):
    return render(request, 'master/index.html')


def list_master_varian(request):
    return render(request, 'master/index_master_varian.html')


def list_master_sinkron(request):
    return render(request, 'master/sinkron.html')


def list_master_process(request):
    master_data_list = []
    dfs = None
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
        master_data = MasterData(id, provider_name_master, address, category_1, category_2, telephone, stateId, cityId)
        master_data_list.append(master_data.__dict__)
        df = df.append(pd.Series(
            {'ProviderId': id, 'stateId': stateId, 'cityId': cityId, 'Category_1': category_1, 'Category_2': category_2,
             'PROVIDER_NAME': provider_name_master, 'ADDRESS': address, 'TEL_NO': telephone},
            name=3))

    df.to_excel("master_provider.xlsx", index=False)

    return JsonResponse(master_data_list, safe=False)


def download_master(request):
    file_path = os.getcwd() + "\\master_provider.xlsx"
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
    file_path = os.getcwd() + "\\master_varian_1.xlsx"
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
        df = cache.get('dataset')
        if df is None:
            df = pd.read_excel("dataset_excel_copy.xlsx")
            cache.set('dataset', df)
        dfs_varian = df.groupby('subject')
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
            row = pd.Series(
                {'course_title': master_data.nama_provider + "#" + master_data.alamat, 'alamat': master_data.alamat,
                 'subject': master_data.nama_provider},
                name=3)
            df = df.append(row)
            df.reset_index(drop=True, inplace=True)

            continue

    df.to_excel("dataset_excel_copy.xlsx", index=False)
    return HttpResponse("Tes")


def master_varian_process(request):
    dff = pd.DataFrame()

    find = False
    master_data_list = []
    dfs = None
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
            {'ProviderId': id, 'ProviderType': "Master", 'stateId': stateId, 'cityId': cityId, 'Category_1': category_1,
             'Category_2': category_2,
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
    dfs = None
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
            {'ProviderId': id, 'ProviderType': "Master", 'stateId': stateId, 'cityId': cityId, 'Category_1': category_1,
             'Category_2': category_2,
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
    if request.method == "POST":

        id = request.POST['id']
        if id is not None and id != '':
            item = ItemProvider.objects.get(pk=id)

            if id in provider_dict_item:
                del provider_dict_item[id]
            else:
                provider_dict_item[id] = item

    context = {"provider_name": provider_dict_item, "link_result": "-"}

    # return HttpResponse(context)
    return render(request, 'matching/temporer.html', context=context)


def read_link_result_and_delete_provider_name2(nama_provider, link_result):
    global dfs

    val = (dfs['Nama'].str.lower().eq(nama_provider.lower()))
    rese = dfs[val]

    # if not rese.empty:
    #     print(rese.index.item())
    # print(dfs)
    global deo
    global deoq
    if not rese.empty:
        #

        try:
            deo = dfs.drop(rese.index.item(), inplace=True)

            val = (dw['Nama'].str.lower().eq(nama_provider.lower()))
            reseq = dw[val]
            if not reseq.empty:
                deoq = dw.drop(reseq.index.item(), inplace=True)
                # deoq = dw
                # if (nama_provider == "klinik takenoko sudirman"):
                #     vae = deoq['Nama Provider'].str.strip().str.lower().eq("klinik takenoko sudirman")
                #     print(deoq[vae])
        except Exception as e:
            for x in rese.index.tolist():
                deo = dfs.drop(x, inplace=True)

            pass


def read_link_result_and_delete_provider_name(nama_provider):
    dfs = pd.read_excel(link_result)
    val = (dfs['Provider Name'].eq(nama_provider.upper()))
    rese = dfs[val]
    print("hapus1 " + nama_provider, rese)

    if not rese.empty:
        print("hapus " + nama_provider, rese)
        deo = dfs.drop(rese.index.item())
        deo.to_excel(link_result, sheet_name='Sheet1', index=False)
        dat = Provider.objects.filter(file_location_result__contains=link_result.split("/")[1]).values()
        dw = pd.read_excel(dat[0]["file_location"])
        val = (dw['Nama Provider'].eq(nama_provider.upper()))
        reseq = dw[val]
        if not reseq.empty:
            deoq = dw.drop(reseq.index.item())
            deoq.to_excel(dat[0]["file_location"], sheet_name='Sheet1', index=False)


def loop_delete(link_result):
    print("loop data2 ", link_result)
    global dfs
    global deo
    global deoq
    global dw
    deo = None
    deoq = None

    file_master = "Master_Add.xlsx"
    df_handler.convert_to_dataframe_from_excel(file_master)
    df = df_handler.get_data_frame()

    dat = Provider.objects.filter(file_location_result__contains=link_result.split("/")[1]).values()
    file_location = dat[0]["file_location"]

    df_handler.convert_to_dataframe_from_excel(file_location)
    dw = df_handler.get_data_frame()

    df_handler.convert_to_dataframe_from_excel(link_result)
    dfs = df_handler.get_data_frame()

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        nama_provider = row['provider_name']
        read_link_result_and_delete_provider_name2(nama_provider, link_result)

    dfs.to_excel(link_result, sheet_name='Sheet1', index=False)


def add_master_store(request):
    if request.method == "POST":
        df = pd.read_excel("Master_Add.xlsx")
        post_ide = request.POST["post_idew"]
        nama_provider = post_ide.split("#")[0]
        alamat = post_ide.split("#")[1]
        link_result = request.POST["link_result"]
        val = (df['provider_name'].str.lower().eq(nama_provider.lower()))
        res = df[val]
        # # # kalau kosong alias belum ada nama provider di dalam file master add, maka proses
        if res.empty:
            row = pd.Series({'provider_name': nama_provider, 'alamat': alamat})
            df = df.append(row, ignore_index=True)
            df.to_excel("Master_Add.xlsx", index=False)

        read_link_result_and_delete_provider_name(nama_provider)



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
        context = {"provider_name": post_ide, "link_result": link_result}
    else:
        if name in provider_liste:
            provider_liste.remove(name)
        context = {"provider_name": provider_liste, "link_result": link_result}
    return render(request, 'matching/temporer.html', context=context)


def add_to_dataset(request):
    if request.method == "POST":
        # OPEN DATASET FILE
        df = dataset.get_bulk_dataset()

        df_basket = pd.read_excel("basket_provider.xlsx")

        # SEARCH PROVIDER IN DATASET
        for label_name, key_provider in list(
                zip(request.POST.getlist('nama_label'), request.POST.getlist('value_provider'))):
            label_name = label_name.split("#")[0]

            item_provider = ItemProvider.objects.get(pk=key_provider)


            for x in range(10):
                try:
                    row = pd.Series({'course_title': item_provider.get_nama_alamat(), 'alamat':item_provider.get_alamat(), 'subject': label_name}, name=3)
                    df = df.append(row, ignore_index=True)
                    cache.delete('dataset')
                except:
                    break



            try:
                rowe = pd.Series({'course_title': item_provider.get_nama_provider(), 'alamat': item_provider.get_alamat()}, name=3)
                df_basket = df_basket.append(rowe, ignore_index=True)
            except:
                break

        # df = df.reset_index(drop=True)
        df_basket.to_excel("basket_provider.xlsx", index=False)
        df.to_excel("dataset_excel_copy.xlsx", index=False)
        # create_model(df)

        context = {"list_pembanding": []}

        return render(request, 'home.html', context)

    return HttpResponse("Marco Polo")


def process_temporer_store(request):


    # dfs = cache.get('dataset')
    # if dfs is None:
    #     dfs = pd.read_excel("dataset_excel_copy.xlsx")
    #     cache.set('dataset', dfs)
    #
    # dfz = dfs.dropna(subset="alamat")
    # dfa = dfz.drop_duplicates(subset='subject')
    label_list = []
    dfa = dataset.get_dataframe_after_cleaned_no_duplicate()
    for index, row in dfa.iterrows():
        alamat = str(row['alamat'])
        label = row["subject"]
        if label + "#" + alamat not in label_list:
            label_list.append(label + "#" + alamat)
    context = {"label_list": label_list, "list": provider_dict_item, "link_result": "-"}
    # return HttpResponse("Process Temporer")
    return render(request, 'matching/proses_temporer.html', context=context)


def get_label(request):
    dfs = cache.get('dataset')
    if dfs is None:
        dfs = pd.read_excel("dataset_excel_copy.xlsx")
        cache.set('dataset', dfs)
    dfs = dfs.sort_values(by=['subject'], ascending=True)
    dfz = dfs.dropna(subset="alamat")
    dfa = dfz.drop_duplicates(subset='subject')
    label_list = []
    print(dfa.size, dfs.size)
    for index, row in dfa.iterrows():
        provider_name = row['course_title']
        alamat = str(row['alamat'])
        label = row["subject"]
        if label + "#" + alamat not in label_list:
            label_list.append(label + "#" + alamat)

    context = {"label_list": label_list}
    return JsonResponse(context, safe=False)


def check_header(df):
    header_list = ['Provinsi', 'Kota', 'Nama Provider', 'Alamat']
    df_header_list = list(df.columns.values)
    if df_header_list == header_list:
        return True
    return False


def vectorize_text(text, tfidf_vec):
    # text = "Klinik Ananda"
    my_vec = tfidf_vec.transform([text])
    return my_vec.toarray()


def cacah_dataframe(df):
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


def is_file_with_this_insurance_exists(nama_asuransi):
    mydata = Provider.objects.filter(nama_asuransi__contains=nama_asuransi).order_by('created_at').values()
    return mydata


def update_perbandingan_excel():
    pass


def perbandingan_result(request):
    global uploaded_file
    global contexte
    global perbandingan_model

    df_handler.set_df_dataset(df_non_duplicate)

    # init Result File Object
    file_result = PerbandinganResult()

    if request.method == 'POST':
        # # # REQUEST DARI PROSES FILE
        if not bool(request.FILES.get('perbandinganModel', False)):
            pembanding_model_return = json.loads(request.POST['processed_file'])
            nama_asuransi = pembanding_model_return["nama_asuransi"]
            provider = Provider.get_model_from_filter(nama_asuransi)

        # # # REQUEST DARI UPLOAD FILE
        else:
            # # init file storage object
            file_storage = FileSystemStorage()

            # # init Perbandingan object
            provider = Provider()

            # # get nama asuransi and file request
            data_asuransi = request.POST['insurance_option']
            file = request.FILES['perbandinganModel']

            nama_asuransi = str(data_asuransi).split("#")[0]
            id_asuransi = str(data_asuransi).split("#")[1]
            # save the file to /media/
            c = file_storage.save(file.name, file)

            # get file url
            file_url = file_storage.path(c)

            # set file location and nama_asuransi to Perbandingan object
            provider.set_file_location(file_url)
            provider.set_nama_asuransi_model(nama_asuransi)
            provider.set_id_asuransi_model(id_asuransi)

        # insert pembanding ke DFHandler
        df_handler.set_perbandingan_model(provider)

        # create file result with compared master
        file_result.create_file_result_with_id_master(df_handler)

        file_result.delete_provider_item_hospital_insurances_with_id_insurances(df_handler)
        file_result.insert_into_end_point_andika_assistant_item_provider(df_handler)

    contexte = {"list": []}
    return render(request, 'matching/perbandingan.html', context=contexte)
