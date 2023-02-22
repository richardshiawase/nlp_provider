import multiprocessing
import queue
import re
import shutil
import string
import time
from multiprocessing import Pool, Process, Queue

import numpy as np
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
import django

from nlp_provider.utils import Pembersih

django.setup()
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

import model.models
from . import views, models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Create your views here.
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
        df = pd.read_excel(uploaded_file)
        create_model(df)


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

    print("Split Dataset")
    # # Split our dataset
    x_train,x_test,y_train,y_test = train_test_split(X,ylabels,test_size=0.2,random_state=42)
    #
    #
    # # Build Model
    print("Fit Model")
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
    print("Fit Model")


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
        print("Fit Model")
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
def cacah_dataframe(df,lr):
    split_row_each = 5000
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
    # df_list.append([df_last,lr])
    df_list.append(df_last)
    # pool_handler(df_list)
    # queue_handler(df_list,lr)
    # process_handler(df_list,lr)
    cnt = 0
    for df in df_list:
        biasa_aja(df,lr,cnt)
        cnt+=1




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
    print("Fit Model")
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

def create_model_bc(df):
    print("Create Model")
    lr_model = LogisticRegression(solver='liblinear')
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
    x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=42)
    #
    #
    # # Build Model
    print("Fit Model")
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


def create_model(dfc):
    print("Create Model")
    lr_model = SGDClassifier(loss='modified_huber',learning_rate='constant',n_jobs=-1,random_state=0,eta0=0.1)
    pembersih = Pembersih(dfc)
    df = pembersih._return_df()
    df['clean_course_title'] = df['course_title']



    print("Improt Tfidf")
    Xfeatures = df['clean_course_title']
    ylabels = df['subject']
    tfidf_vec = TfidfVectorizer()
    X = tfidf_vec.fit_transform(Xfeatures.values.astype('U'))
    print("Split Dataset ")

    x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=1)
    # print(X.shape)
    try:
        print("Fit Model")
        lr_model.partial_fit(x_train, y_train, classes=np.unique(ylabels))
        # calibrator = CalibratedClassifierCV(clf, cv='prefit')
        # model = calibrator.fit(X_tr, y_train)
        print(lr_model.score(x_test, y_test))

    except Exception as e:
        print("sumting wonge "+str(e))





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



    ## Making A Single Prediction
    ex = "RS. CIPUTRA CITRA GARDEN CITY"

    def vectorize_text(text):
        my_vec = tfidf_vec.transform([text])
        return my_vec.toarray()

    vectorize_text(ex)
    df.to_excel('wew.xlsx')
    sample1 = vectorize_text(ex)
    pred = lr_model.predict(sample1)
    #
    print(pred)
    print("Finish Creating Model")



def upload_file_train(request):
    df = pd.read_excel("dataset_excel_copy.xlsx")
    create_model(df)


    list_model = models.Provider_Model.objects.all()
    context_all = {'list_model': list_model}

    return render(request, 'model/model.html', context=context_all)

