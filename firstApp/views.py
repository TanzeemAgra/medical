from django.shortcuts import render
import joblib
import prediction_service
#from webapp.firstApp.models import medical
import yaml
import os
import json
import pandas as pd
import numpy as np
from .models import medical
import psycopg2
import math

# Create your views here.

#cls=joblib.load('../prediction_service/model/model.joblib')

def index(request):
    return render(request, "index.html")

def result(request):
    cls=joblib.load('prediction_service/model/model.joblib')
    lis=[]

    lis.append(float(request.GET['age']))
    lis.append(float(request.GET['sex']))
    lis.append(float(request.GET['bmi']))
    lis.append(float(request.GET['children']))
    lis.append(float(request.GET['smoker']))
    lis.append(float(request.GET['region']))

    answer= cls.predict([lis]).tolist()[0]

    b=medical(age=request.GET['age'],sex=request.GET['sex'],bmi=request.GET['bmi'],children=request.GET['children'],smoker=request.GET['smoker'],region=request.GET['region'],charges=answer)
    b.save()

    return render(request, "index.html",{'answer':answer})