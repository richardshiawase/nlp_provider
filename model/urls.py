from django.urls import path

from . import views

app_name = "model"

urlpatterns = [
    path('',views.index,name="model_route"),
    path('create-model/',views.upload_file,name="create_model"),
    path('train-model/',views.upload_file_train,name="train_model")

]