# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:08:25 2020

@author: hp
"""

import requests
url = 'http://localhost:8501/api'
r = requests.post(url,json={'exp':1.8,})
print(r.json())