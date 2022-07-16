from django.shortcuts import render
import joblib
import prediction_service
import yaml
import os
import json
import joblib
import numpy as np

# Create your views here.

#cls=joblib.load('../prediction_service/model/model.joblib')

def index(request):
    return render(request, "index.html")

def result(request):
    cls=joblib.load('../prediction_service/model/model.joblib')
    lis=[]

    lis.append(float(request.GET['age']))
    lis.append(float(request.GET['sex']))
    lis.append(float(request.GET['bmi']))
    lis.append(float(request.GET['children']))
    lis.append(float(request.GET['smoker']))
    lis.append(float(request.GET['region']))

    answer= cls.predict([lis]).tolist()[0]
    return render(request, "index.html",{'answer':answer})