import json

import requests
from django.http import HttpResponse
from django.shortcuts import render
from requests import Response
from django.http import JsonResponse

# Create your views here.
def index(request):
    # response = requests.get('https://asateknologi.id/api/insuranceall')
    response = requests.get('https://asateknologi.id/api/testapi')
    geodata = response.json()
    geodatas = {'name':'Sol'}
    # return render(request, 'asuransi_view.html', {
    #     'ip': geodata['success'],
    #
    # })
    # print(geodata)

    # return HttpResponse(geodata.get("success"))
    # return render(request,'asuransi_view.html',geodata)