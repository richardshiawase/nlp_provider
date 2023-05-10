"""nlp_provider URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path,include
from . import views

from django.views.generic.base import TemplateView # new

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/",include("django.contrib.auth.urls")),
    path('', views.index, name='home'),  # new
    path('kompilasi', views.kompilasi, name='kompilasi'),  # new
    path('kompilasi_data', views.kompilasi_data, name='kompilasi_data'),  # new

    path('new', views.newe, name='newe'),  # new
    path('master-linked-load', views.master_linked_load, name='master_linked_load'),  # new


    path('perbandingan-upload-page/', views.perbandingan_upload_page, name="perbandingan-upload-page"),

    path('open_file_perbandingan/', views.open_file_perbandingan, name="open_file_perbandingan"),
    path('hapus_tampungan',views.hapus_tampungan,name="hapus_tampungan"),
    path('tampungan_rev/', views.tampungan_rev, name="tampungan_rev"),

    path('match/',views.perbandingan,name="perbandingan"),
    path('match-page/', views.perbandingan_page, name="perbandingan-page"),

    path('hos_ins_list/', views.hos_ins_list, name="hos_ins_list"),
    path('hos_ins_list_page/', views.hos_ins_list_page, name="hos_ins_list_page"),
    path('hos_ins_list_item/', views.hos_ins_list_item, name="hos_ins_list_item"),

    path('tampungan/', views.tampungan, name="tampungan"),
    path('linked-master/', views.linked_master, name="linked-master"),

    path('master/', views.upload_master, name="upload_master"),
    path('master/list', views.list_master, name="master"),
    path('master/list_varian_master', views.list_master_varian, name="master_varian_list"),

    path('master/sinkron', views.list_master_sinkron, name="sinkron_master"),
    path('master/master-varian-process', views.master_varian_process, name="master_varian_process"),
    path('master/sinkron-dataset-process', views.sinkron_dataset_process, name="sinkron_dataset_process"),

    path('master/master-varian-list-read', views.master_varian_list_read, name="master_varian_list_read"),

    path('master/master-list', views.list_master_process, name="master_liste"),
    path('master/master-add', views.master_add, name="master-add"),

    path('master/sinkron-process', views.sinkron_master_process, name="sinkron_master_process"),
    path('master/download_master', views.download_master, name="download_master"),
    path('master/download_master_varian', views.download_master_varian, name="download_master_varian"),

    path('result/', views.perbandingan_result, name="result"),
    path('unlinkhos/', views.unlink_hos, name="unlinkhos"),

    # add to master by dashboard
    path('master/master-add-item', views.add_master_by_dashboard, name="master-add-item"),

    path('temporer-store/',views.temporer_store,name="temporer"),
    path('add-to-master/', views.add_master_store, name="add-to-master"),

    path('update-temporer/', views.temporer_store, name="update_temporer"),
    path('process-temporer/', views.process_temporer_store, name="process_temporer"),
    path('get-label',views.get_label,name="get-label"),
    path('add_to_dataset/',views.add_to_dataset,name="add_to_dataset"),
    # path('tampil_asuransi/', include('tampil_asuransi.urls')),
    path("training/",include("training.urls")),
    path("model/",include("model.urls"))

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
