import re
import shutil
import string

import numpy as np
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

import model.models
from . import views, models


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




def create_model(df):
    # print(df[['course_title','subject']])
    df['clean_course_title'] = df['course_title'].astype(str)
    # df['clean_course_title'] = df['clean_course_title'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if word not in set(all_stopwords)]))
    df['clean_course_title'] = df['clean_course_title'].fillna('').astype(str).replace('', np.nan, regex=False)
    new_string = df['clean_course_title'].str.replace('.','')
    new_string = new_string.str.lower()
    new_string = new_string.str.replace('&','')
    # new_string = new_string.str.replace('-','')

    df['clean_course_title'] = new_string
    print(df)
    # regex = re.compile('[^a-zA-Z]')
    # df['clean_course_title'] = regex.sub('',df['clean_course_title'])

    # print(df[['clean_course_title','course_title']])
    #
    from sklearn.feature_extraction.text import TfidfVectorizer
    Xfeatures = df['clean_course_title']
    ylabels = df['subject']
    tfidf_vec = TfidfVectorizer()
    X = tfidf_vec.fit_transform(Xfeatures.values.astype('U'))
    df_vec = pd.DataFrame(X.todense(),columns=tfidf_vec.get_feature_names_out())
    # print(df_vec.T)

    # # Split our dataset
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(X,ylabels,test_size=0.2,random_state=42)
    #
    #
    # # Build Model
    lr_model = LogisticRegression()
    lr_model.fit(x_train,y_train)
    #
    print(lr_model.score(x_test,y_test))

    pickle.dump(tfidf_vec, open('tfidf_vec.pickle', 'wb'))

    # # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(lr_model, open(filename, 'wb'))


    model_create = models.Provider_Model(model_name=filename, accuracy_score=str(lr_model.score(x_test,y_test)), model_location='drive C')
    model_create.save()

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)
    from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix

    y_pred = lr_model.predict(x_test)



    # Confusion Matrix : true pos,false pos,etc
    print(confusion_matrix(y_pred,y_test))
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

def upload_file_train(request):
    df = pd.read_excel("dataset_excel_copy.xlsx")
    create_model(df)


    list_model = models.Provider_Model.objects.all()
    context_all = {'list_model': list_model}

    return render(request, 'model/model.html', context=context_all)

