# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:41:29 2020

@author: Lenovo
"""
import tkinter as tk
import numpy as np

import pandas as pd

window=tk.Tk()
window.title("health prediction")
window.configure(background="green")
window.geometry('1000x720')
window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)

Lab = tk.Label(window, text = 'health care', bg = 'grey', fg = 'white', font = ('times', 30 , 'italic bold underline'))
Lab.place(x=350, y = 20)

age = tk.Label(window, text = 'Enter age :',font=(20), bg = 'black', fg = 'white')
age.place(x= 300, y = 100)
hage = tk.StringVar()
ageEntry = tk.Entry(window, textvariable = hage).place(x= 450 , y = 100)


gen = tk.Label(window, text = 'Enter gender :',font=(20), bg = 'black', fg = 'white')
gen.place(x= 300, y = 150)
hgen = tk.StringVar()
hgenEntry = tk.Entry(window, textvariable = hgen).place(x= 450 , y = 150)

bps= tk.Label(window, text = 'Enter bps :',font=(20), bg = 'black', fg = 'white')
bps.place(x= 300, y = 200)
hbps = tk.StringVar()
bpsEntry = tk.Entry(window, textvariable = hbps).place(x= 450 , y = 200)

res = tk.Label(window, text = 'Enter results :',font=(20), bg = 'black', fg = 'white')
res.place(x= 300, y = 250)
hres = tk.StringVar()
resEntry = tk.Entry(window, textvariable = hres).place(x= 450 , y = 250)


                
def health():
#path=("/Users/shushruthsheshadri/Desktop/All Files/Project")

#headernames = ['age', 'gender', 'bpm', 'result']

    dataset = pd.read_csv('data4.csv')
    dataset.head()
    
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 3].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 8)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    print(y_pred)
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    result = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(result)
    result1 = classification_report(y_test, y_pred)
    print("Classification Report:",)
    print (result1)
    result2 = accuracy_score(y_test,y_pred)
    print("Accuracy:",result2)
       
    x1 = str(hage.get())
    print('age:'+x1)
        
    x2 = str(hgen.get())
    print('gender:'+x2)
        
    x3 = str(hbps.get())
    print('bps:'+x3)
        
    x4 = str(hres.get())
    print('result:'+x4)
    
    
    if (int(x4)==1):
        hres.set('Athlete')
        print('Athlete')
    elif (int(x4)==2):
        hres.set('Excellent')
        print('Excellent')
    elif (int(x4)==3):
        hres.set('Great')
        print('Great')
    elif (int(x4)==4):
        hres.set('Good')
        print('Good')
    elif (int(x4)==5):
        hres.set('Average')
        print('Average')
    elif (int(x4)==6):
        hres.set(' Below Average')
        print(' Below Average')
    else: 
        hres.set('very poor')
        print('very poor')
    
pred = tk.Button(window, text="health prediction", command = health, fg = 'white', bg= 'black', width = 20, height = 1, activebackground= 'Red', font=('times', 30, 'bold'))
pred.place(x = 400,y = 500)
    
window.mainloop()

'''
window=tk.Tk()
window.title("health prediction")
window.configure(background="green")
window.geometry('1000x720')
window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)

Lab = tk.Label(window, text = 'health care', bg = 'blue', fg = 'white', font = ('times', 30 , 'italic bold underline'))
Lab.place(x=350, y = 20)

age = tk.Label(window, text = 'Enter age :',font=(20), bg = 'orange', fg = 'black')
age.place(x= 300, y = 100)
hage = tk.StringVar()
ageEntry = tk.Entry(window, textvariable = hage).place(x= 450 , y = 100)


gen = tk.Label(window, text = 'Enter bps :',font=(20), bg = 'orange', fg = 'black')
gen.place(x= 300, y = 150)
hgen = tk.StringVar()
hgenEntry = tk.Entry(window, textvariable = hgen).place(x= 450 , y = 150)

bps= tk.Label(window, text = 'Enter gender :',font=(20), bg = 'orange', fg = 'black')
bps.place(x= 300, y = 200)
hbps = tk.StringVar()
bpsEntry = tk.Entry(window, textvariable = hbps).place(x= 450 , y = 200)

res = tk.Label(window, text = 'Enter results :',font=(20), bg = 'orange', fg = 'black')
res.place(x= 300, y = 250)
hres = tk.StringVar()
resEntry = tk.Entry(window, textvariable = hres.place(x= 450 , y = 250)
'''






