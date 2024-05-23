from django.shortcuts import render
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import math
import plotly.express as px

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    merged = pd.read_csv(r'C:\Users\gupta\Desktop\gautam4.csv')
    X = merged.drop(columns=['price'], axis=1)
    y = merged['price']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    model=RandomForestRegressor()
    model.fit(x_train, y_train)
    # predictions = model.predict(x_test)

    seller_type = int(request.POST.get('sellerType'))
    bedroom = int(request.POST.get('bedroom'))
    layout_type = int(request.POST.get('layoutType'))
    property_type = int(request.POST.get('propertyType'))
    area = int(request.POST.get('area'))
    furnish_type = int(request.POST.get('furnishType'))
    bathroom = int(request.POST.get('bathroom'))
    city = int(request.POST.get('city'))
    st = int(request.POST.get('st'))
    input_data = pd.DataFrame({
        'seller_type': [seller_type],
        'bedroom': [bedroom],
        'layout_type': [layout_type],
        'property_type': [property_type],
        'area': [area],
        'furnish_type': [furnish_type],
        'bathroom': [bathroom],
        'city': [city],
        'st': [st]
    })

    predicted = model.predict(input_data)
    predicted = round(predicted[0])
    rate = predicted % 100
    if rate < 50:
        lower = predicted - rate
        upper = predicted + (50 - rate)
    else:
        lower = predicted + (50 - rate)
        upper = predicted + (100 - rate)

    price = "The predicted range rent  is &#8377; {}-{}".format(lower, upper)
    return render(request, "predict.html", {"result2":price})
