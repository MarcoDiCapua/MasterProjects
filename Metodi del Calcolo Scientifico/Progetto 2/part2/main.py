import numpy as np
import time
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI
import PySimpleGUI as sg
from PIL import Image
from utils import *

def show_windows(im):
    image_el = [sg.Image(im, expand_x=True, expand_y=True)]
    layout_2 = [ [image_el],
                 [sg.Button("Another"), sg.Button("Cancel")]]
    window2 = sg.Window('Window Title', layout_2).finalize()
    window2.Maximize()
    while True:
        event, values = window2.read()
        if event == "Another":
            window2.close()
            #torna 1 e cancella vecchia finestra
            return 1
        if event == "Cancel" or event == sg.WIN_CLOSED:
            window2.close()
            #torna zero e chiude tutto
            return 0

def main():
    img = ""
    f = ""
    d = ""
    sg.theme('Black')   # Add a touch of color
    # All the stuff inside your window.
    layout = [  [sg.Text("Inserisci una immagine .bmp, in scala di grigi")],
                [sg.Text("Inserisci un intero F minore della metà dimensione più piccola dell'immagine, sarà la finestra per DTC2")],
                [sg.Text("Inserisci un intero d tra 0 e 2F-2, sarà la finestra per DTC2soglia di taglio delle frequenze")],
                [sg.Text('Inserisci una immagine'), sg.In(size=(25, 1), enable_events=True, key='-IMG-'), sg.FileBrowse(), sg.Text('',visible=False, key="-DIM-")],
                [sg.Text('Inserisci F'), sg.In(size=(25, 1), enable_events=True, key='-F-'), sg.Button("Enter", key="-ENTER_F-", disabled=True), sg.Text('',visible=False, key="-F_CHECK-")],
                [sg.Text('Inserisci d'), sg.In(size=(25, 1), enable_events=True, key='-D-'), sg.Button("Enter", key="-ENTER_D-", disabled=True), sg.Text('',visible=False, key="-D_CHECK-")],
                [sg.Button('Ok', disabled=True), sg.Button('Cancel')]]
    
    # Create the Window
    window = sg.Window('Window Title', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == "-IMG-":
            [img, x] = check_image(values["-IMG-"])
            if x == 1:
                window["-DIM-"].update("Questo file non è una immagine")
                window["-DIM-"].update(visible=True)
            if x == 2:
                window["-DIM-"].update("L'immagine non è .bmp")
                window["-DIM-"].update(visible=True)
            if x == 3:
                window["-DIM-"].update("L'immagine non è in scala di grigi")
                window["-DIM-"].update(visible=True)
            if x == 0:
                window["-DIM-"].update(img.size)
                window["-DIM-"].update(visible=True)
                window["-ENTER_F-"].update(disabled=False)
        if event == "-ENTER_F-":
            f = values["-F-"]
            w,h = img.size
            if not f.isdigit():
                window["-F_CHECK-"].update("Il valore non è intero")
                window["-F_CHECK-"].update(visible=True)
            elif int(f)<=0 or int(f) > w/2 or int(f) > h/2:
                window["-F_CHECK-"].update("Il valore è troppo grande o inferiore a 0")
                window["-F_CHECK-"].update(visible=True)
            else:
                f = int(f)
                window["-F_CHECK-"].update("OK")
                window["-F_CHECK-"].update(visible=True)
                window["-ENTER_D-"].update(disabled=False)
        if event == "-ENTER_D-":
            d = values["-D-"]
            if not d.isdigit():
                window["-D_CHECK-"].update("Il valore non è intero")
                window["-D_CHECK-"].update(visible=True)
            elif int(d)<0 or int(d)>(2*f-2):
                window["-D_CHECK-"].update("Il valore non è compreso tra 0 e 2F-2")
                window["-D_CHECK-"].update(visible=True)
            else:
                d = int(d)
                window["-D_CHECK-"].update("OK")
                window["-D_CHECK-"].update(visible=True)
                window["Ok"].update(disabled=False)
        if event == "Ok":
            image_compress(values['-IMG-'], f, d)
            window.hide()
            b = show_windows('./out.png')
            
            if b == 0:
                window.un_hide()
                window.close()
            if b == 1:
                window["-IMG-"].update('')
                window["-DIM-"].update('')
                window["-F-"].update('')
                window["-D-"].update('')
                window["-ENTER_F-"].update(disabled = True)
                window["-ENTER_D-"].update(disabled = True)
                window["-D_CHECK-"].update(visible = False)
                window["-F_CHECK-"].update(visible = False)
                window["Ok"].update(disabled = True)
                window.un_hide()
        if event == "Cancel" or event == sg.WIN_CLOSED:
            break

    window.close()
            

if __name__ == '__main__':
    main()
