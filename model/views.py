import multiprocessing
import queue
import re
import shutil
import string
import sys
import time
from multiprocessing import Pool, Process, Queue

from django.core.cache import cache
from numba import jit, cuda
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import dask
import dask.dataframe as dd
import dask.delayed as delayed
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
import django
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, scale

django.setup()
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import model.models
from . import views, models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Create your views here.
x_train=0
x_test=0
y_train=0
y_test=0

def index(request):
    list_model = models.Provider_Model.objects.all()
    context_all = {'list_model':list_model}
    return render(request,"model/model.html",context=context_all)


def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['documentModel']
        print(uploaded_file.name)
        fs = FileSystemStorage()
        fs.save(uploaded_file.name,uploaded_file)
        # Load Dataset
        df1 = pd.read_excel(uploaded_file)
        # df = df1.drop_duplicates(keep='first')

        # parts = dask.delayed(pd.read_excel)(uploaded_file, sheet_name=0, usecols=[0,1,2])
        # df1 = dd.from_delayed(parts)

        # # # Split dataframe to many
        # df_list = cacah_dataframe(df)
        create_model(df1)


    list_model = models.Provider_Model.objects.all()
    context_all = {'list_model':list_model}

    return render(request, 'model/model.html',context=context_all)

def work_log(work_data):
    df = work_data[0]
    lr_model = work_data[1]
    print(df)
    # print(" Process %s waiting %s seconds" % (work_data[0].head(), work_data[1]))
    # print(" Process %s Finished." % work_data[0].head())
    #
    df['clean_course_title'] = df['course_title'].astype(str)
    # df['clean_course_title'] = df['clean_course_title'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if word not in set(all_stopwords)]))
    df['clean_course_title'] = df['clean_course_title'].fillna('').astype(str).replace('', np.nan, regex=False)
    new_string = df['clean_course_title'].str.replace('.','')
    new_string = new_string.str.lower()
    new_string = new_string.str.replace('&','')
    df['clean_course_title'] = new_string
    print("Improt Tfidf")
    Xfeatures = df['clean_course_title']
    ylabels = df['subject']
    tfidf_vec = TfidfVectorizer()
    X = tfidf_vec.fit_transform(Xfeatures.values.astype('U'))
    Y = tfidf_vec.fit_transform(ylabels.values.astype('U'))

    print("Split Dataset")
    # # Split our dataset
    import sklearn.model_selection as ms
    x_train,x_test,y_train,y_test = ms.train_test_split(X,Y,test_size=0.2,random_state=42)
    #
    #
    # # Build Model
    print("Fit Modelz")
    lr_model.fit(x_train,y_train)
    #
    print(lr_model.score(x_test,y_test))

    print("Open Pickle")
    pickle.dump(tfidf_vec, open('tfidf_vec.pickle', 'wb'))

    # # save the model to disk
    print("Save model to disk")
    filename = 'finalized_model.sav'
    pickle.dump(lr_model, open(filename, 'wb'))

    print("Save Model")
    model_create = models.Provider_Model(model_name=filename, accuracy_score=str(lr_model.score(x_test,y_test)), model_location='drive C')
    model_create.save()

    # load the model from disk
    print("Load Model")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)

    ex = "RS. CIPUTRA CITRA GARDEN CITY"


    def vectorize_text(text):
        my_vec = tfidf_vec.transform([text])
        return my_vec.toarray()

    vectorize_text(ex)
    # df.to_excel('wew.xlsx')
    sample1 = vectorize_text(ex)
    pred = lr_model.predict(sample1)

    print(pred)
    print("Finish Creating Model")


def work_log_process(df_to_accomplish,lr_model):
    df = df_to_accomplish
    # print(" Process %s waiting %s seconds" % (work_data[0].head(), work_data[1]))
    # print(" Process %s Finished." % work_data[0].head())
    #
    df['clean_course_title'] = df['course_title'].astype(str)
    # df['clean_course_title'] = df['clean_course_title'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if word not in set(all_stopwords)]))
    df['clean_course_title'] = df['clean_course_title'].fillna('').astype(str).replace('', np.nan, regex=False)
    new_string = df['clean_course_title'].str.replace('.', '')
    new_string = new_string.str.lower()
    new_string = new_string.str.replace('&', '')
    df['clean_course_title'] = new_string
    print("Improt Tfidf")
    Xfeatures = df['clean_course_title']
    ylabels = df['subject']
    tfidf_vec = TfidfVectorizer()
    X = tfidf_vec.fit_transform(Xfeatures.values.astype('U'))

    print("Split Dataset")
    # # Split our dataset
    x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)
    #
    #
    filename = 'finalized_model.sav'
    # open saved model
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    # print(result)
    # # Build Model
    # print("Fit Model")
    # lr_model.fit(x_train, y_train)
    #
    # #
    # print(lr_model.score(x_test, y_test))
    #
    # print("Open Pickle")
    # pickle.dump(tfidf_vec, open('tfidf_vec.pickle', 'wb'))
    #
    # # # save the model to disk
    # print("Save model to disk")
    # pickle.dump(lr_model, open(filename, 'wb'))
    #
    # print("Save Model")
    # model_create = models.Provider_Model(model_name=filename,
    #                                      accuracy_score=str(lr_model.score(x_test, y_test)),
    #                                      model_location='drive C')
    # model_create.save()
    #
    # # load the model from disk
    # print("Load Model")
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(x_test, y_test)
    # print(result)
    #
    # ex = "RS. CIPUTRA CITRA GARDEN CITY"
    #
    # def vectorize_text(text):
    #     my_vec = tfidf_vec.transform([text])
    #     return my_vec.toarray()
    #
    # vectorize_text(ex)
    # # df.to_excel('wew.xlsx')
    # sample1 = vectorize_text(ex)
    # pred = lr_model.predict(sample1)
    #
    # print(pred)
    # print("Finish Creating Model")
    return True

def vectorize_text(text,tfidf_vec):
        my_vec = tfidf_vec.transform([text])
        return my_vec.toarray()

def biasa_aja(df,lr_model,cnt):
    # # Build Model
    filename = 'finalized_model_'+str(cnt)+'.sav'
    pickle_name = 'tfidf_vec.pickle'
    df['clean_course_title'] = df['course_title'].astype(str)
    # df['clean_course_title'] = df['clean_course_title'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if word not in set(all_stopwords)]))
    df['clean_course_title'] = df['clean_course_title'].fillna('').astype(str).replace('', np.nan, regex=False)
    new_string = df['clean_course_title'].str.replace('.', '')
    new_string = new_string.str.lower()
    new_string = new_string.str.replace('&', '')
    df['clean_course_title'] = new_string
    print(df.shape)
    print("Improt Tfidf")
    Xfeatures = df['clean_course_title']
    ylabels = df['subject']

    # buka file
    try:
        loaded_model = pickle.load(open(filename, 'rb'))
        tfidf_vec = pickle.load(open(pickle_name,'rb'))
        lr_model = loaded_model

    except:
        print("gagal")
        tfidf_vec = TfidfVectorizer()

    X = tfidf_vec.fit_transform(Xfeatures.values.astype('U'))
    print("Split Dataset")
    # # Split our dataset
    x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

    lr_model.fit(x_train, y_train)

    print("Open Pickle")
    pickle.dump(tfidf_vec, open(pickle_name, 'wb'))

    # # save the model to disk
    print("Save model to disk")
    filenamez = "finalized_model"+"_"+str(cnt)+".sav"
    pickle.dump(lr_model, open(filenamez, 'wb'))

    print("Save Model")
    model_create = models.Provider_Model(model_name=filenamez,
                                         accuracy_score=str(lr_model.score(x_test, y_test)),
                                         model_location='drive C')
    model_create.save()

    # load the model from disk
    print("Load Model")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)
    ex = "RS. CIPUTRA CITRA GARDEN CITY"
    vectorize_text(ex,tfidf_vec)
    # df.to_excel('wew.xlsx')
    sample1 = vectorize_text(ex,tfidf_vec)
    pred = lr_model.predict(sample1)

    print(pred)
    print("Finish Creating Model")
    print("Fit Model1")


def pool_handler(df_list):
    p = Pool(2)
    p.map(work_log,df_list)

def queue_handler(df_list,lr_model):
    cnt = 1
    que = Queue()
    for df in df_list:
        que.put(df)

    while not que.empty():
        df = que.get()
        print(df)
        df['clean_course_title'] = df['course_title'].astype(str)
        # df['clean_course_title'] = df['clean_course_title'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if word not in set(all_stopwords)]))
        df['clean_course_title'] = df['clean_course_title'].fillna('').astype(str).replace('', np.nan, regex=False)
        new_string = df['clean_course_title'].str.replace('.', '')
        new_string = new_string.str.lower()
        new_string = new_string.str.replace('&', '')
        df['clean_course_title'] = new_string
        print("Improt Tfidf")
        Xfeatures = df['clean_course_title']
        ylabels = df['subject']
        tfidf_vec = TfidfVectorizer()
        X = tfidf_vec.fit_transform(Xfeatures.values.astype('U'))

        print("Split Dataset")
        # # Split our dataset
        x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)
        #
        #
        # # Build Model
        print("Fit Model2")
        lr_model.fit(x_train, y_train)
        #
        print(lr_model.score(x_test, y_test))

        print("Open Pickle")
        pickle.dump(tfidf_vec, open('tfidf_vec.pickle', 'wb'))

        # # save the model to disk
        print("Save model to disk")
        filename = 'finalized_model.sav'
        pickle.dump(lr_model, open(filename, 'wb'))

        print("Save Model")
        model_create = models.Provider_Model(model_name=filename,
                                             accuracy_score=str(lr_model.score(x_test, y_test)),
                                             model_location='drive C')
        model_create.save()

        # load the model from disk
        print("Load Model")
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(x_test, y_test)
        print(result)

        ex = "RS. CIPUTRA CITRA GARDEN CITY"

        def vectorize_text(text):
            my_vec = tfidf_vec.transform([text])
            return my_vec.toarray()

        vectorize_text(ex)
        # df.to_excel('wew.xlsx')
        sample1 = vectorize_text(ex)
        pred = lr_model.predict(sample1)

        print(pred)
        print("Finish Creating Model")

def process_handler(df_list,lr_model):
    number_of_process = len(df_list)
    df_to_accomplish = Queue()
    for df in df_list:
        df_to_accomplish.put(df)

    processes = []
    for x in range(number_of_process):
        p = Process(target=work_log_process,args=(df_list[x],lr_model))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
def cacah_dataframe(df):
    split_row_each = 970000
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
    # split_row_each = 5000
    # start_index = 0
    # iteration_count = int(df.shape[0]/split_row_each)
    # sisa = df.shape[0]%split_row_each
    # sisa_row = iteration_count*split_row_each+sisa
    # df_list = []
    # for x in range(iteration_count):
    #     end_index = start_index + split_row_each
    #     df_new= df.iloc[start_index:end_index]
    #     start_index = end_index
    #     # df_list.append([df_new,lr])
    #     df_list.append(df_new)
    # aw = lambda x,y : y if x > 0 else 0
    # df_last = df.iloc[start_index:aw(sisa, sisa_row)]
    # # df_list.append([df_last,lr])
    # df_list.append(df_last)
    # # pool_handler(df_list)
    # # queue_handler(df_list,lr)
    # # process_handler(df_list,lr)
    # cnt = 0
    # for df in df_list:
    #     biasa_aja(df,lr,cnt)
    #     cnt+=1



def process_partially(df):
    print("Number of cpu : ",multiprocessing.cpu_count())

    df['clean_course_title'] = df['course_title'].astype(str)
    # df['clean_course_title'] = df['clean_course_title'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if word not in set(all_stopwords)]))
    df['clean_course_title'] = df['clean_course_title'].fillna('').astype(str).replace('', np.nan, regex=False)
    new_string = df['clean_course_title'].str.replace('.', '')
    new_string = new_string.str.lower()
    new_string = new_string.str.replace('&', '')
    # new_string = new_string.str.replace('-','')
    df['clean_course_title'] = new_string
    # regex = re.compile('[^a-zA-Z]')
    # df['clean_course_title'] = regex.sub('',df['clean_course_title'])

    # print(df[['clean_course_title','course_title']])
    #
    print("Improt Tfidf")
    Xfeatures = df['clean_course_title']
    ylabels = df['subject']
    tfidf_vec = TfidfVectorizer()
    X = tfidf_vec.fit_transform(Xfeatures.values.astype('U'))

    print("Split Dataset")
    # # Split our dataset
    x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)
    #
    #
    # # Build Model
    lr_model = LogisticRegression(warm_start=True)
    print("Fit Model3")
    lr_model.fit(x_train, y_train)
    #
    print(lr_model.score(x_test, y_test))

    print("Open Pickle")
    pickle.dump(tfidf_vec, open('tfidf_vec.pickle', 'wb'))

    # # save the model to disk
    print("Save model to disk")
    filename = 'finalized_model.sav'
    pickle.dump(lr_model, open(filename, 'wb'))

    print("Save Model")
    model_create = models.Provider_Model(model_name=filename, accuracy_score=str(lr_model.score(x_test, y_test)),
                                         model_location='drive C')
    model_create.save()

    # load the model from disk
    print("Load Model")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)
    from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

    y_pred = lr_model.predict(x_test)

    # Confusion Matrix : true pos,false pos,etc
    # print(confusion_matrix(y_pred,y_test))
    # print(df['subject'].unique())
    # print(classification_report(y_pred,y_test))
    # plot_confusion_matrix(lr_model,x_test,y_test)

    ### Making A Single Prediction
    ex = "RS. CIPUTRA CITRA GARDEN CITY"

    def vectorize_text(text):
        my_vec = tfidf_vec.transform([text])
        return my_vec.toarray()

    vectorize_text(ex)
    df.to_excel('wew.xlsx')
    sample1 = vectorize_text(ex)
    pred = lr_model.predict(sample1)

    print(pred)
    print("Finish Creating Model")



def create_model(df):
    print("Create Model")
    # lr_model = LogisticRegression(solver='sag',warm_start=True)

    lr_model = SGDClassifier(loss='modified_huber',n_jobs=-1,random_state=0)
    df_list = cacah_dataframe(df)
    i = 0
    tfidf_vec = TfidfVectorizer()
    akurasi = 0
    for df in df_list:
        df['clean_course_title'] = df['course_title'].astype(str)
        # df['clean_course_title'] = df['clean_course_title'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if word not in set(all_stopwords)]))
        df['clean_course_title'] = df['clean_course_title'].fillna('').astype(str).replace('', np.nan, regex=False)
        new_string = df['clean_course_title'].str.replace('.', '')
        new_string = new_string.str.lower()
        new_string = new_string.str.replace('&', '')
        # new_string = new_string.str.replace('-','')
        df['clean_course_title'] = new_string


        print("Improt Tfidf")
        Xfeatures = df['clean_course_title']
        ylabels = df['subject']

        print("Fit Model index "+str(i))
        global x_test
        global y_test
        global classes
        global yp
        if i == 0 :

            X = tfidf_vec.fit_transform(Xfeatures.values.astype('U'))
            # X = scale(Xfeatures)
            # X = Xfeatures.to_numpy()
            # y = ylabels.to_numpy()
            # y = scale(ylabels)
            # y = tfidf_vec.transform(y.values.astype('U'))
            # print(Xfeatures.sort_index(ascending=True))
            # print(Xfeatures.sort_index(ascending=True))
            print("Split Dataset "+str(type(X)))

            x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=1)
            # print(X.shape)
            if(i==0):
                classes = ylabels
                yp = ylabels
            # print(classes)
            # yp = y_train
            # lr_model.fit(X, ylabels, classes=classes)
            try:
                lr_model.partial_fit(x_train, y_train, classes=np.unique(ylabels))
                akurasi = lr_model.score(x_test,y_test)
                print(akurasi)
                # print(lr_model.coef_)
            except Exception as e:
                print("sumting wonge "+str(e))
        elif i > 0 :
            # lr_model1 = LogisticRegression(solver='sag', warm_start=True)

            Xe = tfidf_vec.transform(Xfeatures.values.astype('U'))
            # Y = tfidf_vec.transform(ylabels.values.astype('U'))
            # print(classes)
            print(Xe.shape)

            # if(X.shape[0] != 35):
            #     break

            print("Split Dataset")

            x_traine, x_teste, y_traine, y_test = train_test_split(Xe, ylabels, test_size=0.2, random_state=42)
            print(x_traine,y_traine.shape)
            print(y_traine)
            # print(classes)
            # lr_model.partial_fit(X, yp, classes=None)
            try:
                lr_model.partial_fit(x_traine, y_traine)
                akurasi = lr_model.score(x_teste, y_test)
                print(akurasi)
            except Exception as e:
                print("sumting wonge 2"+str(e))
                # break
            # print(lr_model.coef_)

        #
        i+=1
    ## Making A Single Prediction
    ex = "altis"
    # print(lr_model.coef_)

    def vectorize_text(text):
        my_vec = tfidf_vec.transform([text])
        return my_vec.toarray()

    vectorize_text(ex)
    df.to_excel('wew.xlsx')
    sample1 = vectorize_text(ex)
    pred = lr_model.predict(sample1)

    print(pred)

    print("Open Pickle")
    pickle.dump(tfidf_vec, open('tfidf_vec.pickle', 'wb'))

    # # save the model to disk
    print("Save model to disk")
    filename = 'finalized_model.sav'
    pickle.dump(lr_model, open(filename, 'wb'))
    print("Save Model")
    model_create = models.Provider_Model(model_name=filename, accuracy_score=str(akurasi),

                                         model_location='drive C')
    model_create.save()
        #
    # load the model from disk
    # print("Load Model")
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(x_test, y_test)
    # print(result)
    from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

    y_pred = lr_model.predict(x_test)

    # Confusion Matrix : true pos,false pos,etc
    # print(confusion_matrix(y_pred,y_test))
    # print(df['subject'].unique())
    # print(classification_report(y_pred,y_test))
    # plot_confusion_matrix(lr_model,x_test,y_test)

    ### Making A Single Prediction
    # ex = "RS. CIPUTRA CITRA GARDEN CITY"
    #
    # def vectorize_text(text):
    #     my_vec = tfidf_vec.transform([text])
    #     return my_vec.toarray()
    #
    # vectorize_text(ex)
    # df.to_excel('wew.xlsx')
    # sample1 = vectorize_text(ex)
    # pred = lr_model.predict(sample1)
    #
    # print(pred)
    print("Finish Creating Model")

    # cacah_dataframe(df,lr_model)
    # proc = Process(target=process_partially,args=(df,))
    # proc.start()
    # proc.join()

def train_model(df):
    # lr_model = SGDClassifier()
    filename = 'finalized_model.sav'

    lr_model = pickle.load(open(filename, 'rb'))

    print("train model")
    df['clean_course_title'] = df['course_title'].astype(str)
    # df['clean_course_title'] = df['clean_course_title'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if word not in set(all_stopwords)]))
    df['clean_course_title'] = df['clean_course_title'].fillna('').astype(str).replace('', np.nan, regex=False)
    new_string = df['clean_course_title'].str.replace('.', '')
    new_string = new_string.str.lower()
    new_string = new_string.str.replace('&', '')
    # new_string = new_string.str.replace('-','')
    df['clean_course_title'] = new_string
    # regex = re.compile('[^a-zA-Z]')
    # df['clean_course_title'] = regex.sub('',df['clean_course_title'])



    Xfeatures = df['clean_course_title']
    ylabels = df['subject']
    # del df
    # del new_string
    # gc.collect()
    tfidf_vec = TfidfVectorizer()
    X = tfidf_vec.fit_transform(Xfeatures.values.astype('U'))
    # X = tfidf_vec.transform(Xfeatures.values.astype(str))

    print("Split Dataset")

    x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=19)

    #
    #
    # del X
    # del tfidf_vec
    # # Build Model
    print("Fit Model4")
    try:
        # lr_model.n_estimators+=1
        lr_model.partial_fit(x_train, y_train,classes=np.unique(y_train))
    except Exception as e:
        print("Sumting wonge "+str(e))
    #
    print(lr_model.score(x_test, y_test))

    print("Open Pickle")
    pickle.dump(tfidf_vec, open('tfidf_vec.pickle', 'wb'))
    #
    # # # save the model to disk
    print("Save model to disk")
    filename = 'finalized_model.sav'
    pickle.dump(lr_model, open(filename, 'wb'))
    #
    # print("Save Model")
    model_create = models.Provider_Model(model_name=filename, accuracy_score=str(lr_model.score(x_test, y_test)),
                                         model_location='drive C')
    model_create.save()
    #
    # # load the model from disk
    print("Load Model")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)
    # from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
    #
    y_pred = lr_model.predict(x_test)
    #
    # # Confusion Matrix : true pos,false pos,etc
    # # print(confusion_matrix(y_pred,y_test))
    # # print(df['subject'].unique())
    # # print(classification_report(y_pred,y_test))
    # # plot_confusion_matrix(lr_model,x_test,y_test)
    #
    # ### Making A Single Prediction
    ex = "AIS HEALTHCARE"
    #
    def vectorize_text(text):
        my_vec = tfidf_vec.transform([text])
        return my_vec.toarray()
    #
    vectorize_text(ex)
    df.to_excel('wew.xlsx')
    sample1 = vectorize_text(ex)
    pred = lr_model.predict(sample1)
    #
    print(pred)
    print("Finish Creating Model")

    # cacah_dataframe(df,lr_model)
    # proc = Process(target=process_partially,args=(df,))
    # proc.start()
    # proc.join()

def upload_file_train(request):
    df = pd.read_excel("dataset_excel_copy.xlsx")

    # create_model(df)
    train_model(df)

    list_model = models.Provider_Model.objects.all()
    context_all = {'list_model': list_model}

    return render(request, 'model/model.html', context=context_all)

