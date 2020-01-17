<<<<<<< HEAD
#!/usr/bin/env python
import sys
import PySimpleGUI as sg
import os
from sys import exit as exit
import h5py

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import math
import pandas as pd 
from skimage.io import imread
from skimage import io, transform
import cv2
from scipy.io import loadmat
import glob

# for drawing plt results to gui
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.tkagg as tkagg
import tkinter as Tk

from infer import _infer #, _infer_compare


def preprocess(img):
    print('starting preprocess')
    cv2_img = cv2.imread(img)
    b,g,r = cv2.split(cv2_img)

    # applied clahe on 3 channel
    clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(3,3))
    contrast_b = clahe.apply(b)
    contrast_g = clahe.apply(g)
    contrast_r = clahe.apply(r)

    # merge 3 channel
    preprocessed_path = img + '_processed.JPEG'
    print("preprocessed path: " + preprocessed_path)
    contrast = cv2.merge([contrast_b,contrast_g,contrast_r])
    cv2.imwrite(preprocessed_path, contrast)
    return preprocessed_path
    
    print('end preprocess')
      
def predict(img, suffix):
    checkpoint = 'test/checkpoints/model-90000.pth'
    backbone = 'resnet101'
    prob_thres = 0.6
    dataset = 'voc2007'

    _output = img + suffix
    _infer(img,_output,checkpoint,dataset,backbone,prob_thres)
    #_infer_compare(img,_output,checkpoint,dataset,backbone,prob_thres)
    return _output
    
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top')
    return figure_canvas_agg
    
def prepare_figure_preprocessed(image):
    plt.figure(figsize = (8,5))
    #plt.scatter(pred_x, pred_y)
    plt.imshow(image, cmap="gray")
    return plt.gcf()
   

window = sg.Window('Crack/Corrosion Detection', return_keyboard_events=True, location=(0,0), use_default_focus=False, resizable=True)
def launch():
    width = 10
    height = 6

    og_canvas = sg.Canvas(size=(width, height), key='og_canvas')
    preprocessed_canvas = sg.Canvas(size=(width, height), key='preprocessed_canvas')
    predicted_canvas = sg.Canvas(size=(width, height), key='predicted_canvas')
    og_predicted_canvas = sg.Canvas(size=(width, height), key='og_predicted_canvas')
   
    read_frame_layout = [
                        [sg.OK()]
                    ]
    read_frame = sg.Frame('Choose File', read_frame_layout, font='Any 12', title_color='blue')

    og_frame_layout = [[og_canvas],
                   [sg.Text('Raw')],
                   ]
    
    og_frame = sg.Frame('Raw', og_frame_layout, font='Any 12', title_color='blue')
    
    preprocessed_frame_layout = [[preprocessed_canvas],
                   [sg.Text('Preprocessed')],
                   ]
    
    preprocessed_frame = sg.Frame('Preprocessed', preprocessed_frame_layout, font='Any 12', title_color='blue')
   
    predicted_frame_layout = [[predicted_canvas],
                   [sg.Text('predicted')],
                   ] 
    predicted_frame = sg.Frame('Predicted w/ CLAHE', predicted_frame_layout, font='Any 12', title_color='blue')
    
    og_predicted_frame_layout = [[og_predicted_canvas],
                   [sg.Text('predicted w/o CLAHE')],
                   ]
    
    og_predicted_frame = sg.Frame('Predicted w/o CLAHE', og_predicted_frame_layout, font='Any 12', title_color='blue')
    

    layout = [
              [read_frame],
              [og_frame, preprocessed_frame],
              [predicted_frame, og_predicted_frame]
             ]
    
    window.Layout(layout)
    window.Finalize()
    while(True):
        
        image_file = sg.PopupGetFile('Image to open')
        fig_image = prepare_figure_preprocessed(np.asarray(Image.open(image_file)))
        
        if image_file is None:
            print("bye")
            exit(0)
        
        # preprocess image
        preprocessed_path = preprocess(image_file)
        preprocessed_fig_image = prepare_figure_preprocessed(np.asarray(Image.open(preprocessed_path)))

        # predict
        predicted_path = predict(preprocessed_path, '_predicted_w_clahe.JPEG')
        predicted_fig_image = prepare_figure_preprocessed(np.asarray(Image.open(predicted_path)))
       
        predicted_wo_clahe_path = predict(image_file, '_predicted_wo_clahe.JPEG')
        predicted_wo_clahe_fig_image = prepare_figure_preprocessed(np.asarray(Image.open(predicted_wo_clahe_path)))
       

        # define layout, show and read the window
        og_photo = draw_figure(og_canvas.TKCanvas, fig_image)
        preprocessed_photo = draw_figure(preprocessed_canvas.TKCanvas, preprocessed_fig_image)
        predicted_photo = draw_figure(predicted_canvas.TKCanvas, predicted_fig_image)
        predicted_wo_clahe_photo = draw_figure(og_predicted_canvas.TKCanvas, predicted_wo_clahe_fig_image)
        
        #window.Refresh() 
        
        while True:
            event, values = window.read()
            if event == 'Maximize':
                window.Maximize()
            elif event == 'OK':
                print('EKEKEKEKEKE')
                break
            elif event == 'close':
                window.close()
                window.destroy()

if __name__== "__main__":
    launch()
=======
#!/usr/bin/env python
import sys
import PySimpleGUI as sg
import os
from sys import exit as exit
#from keras.models import load_model
#from keras import backend as K
#import tensorflow as tf
import h5py

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import math
import pandas as pd 
from skimage.io import imread
from skimage import io, transform
import cv2
from scipy.io import loadmat
import glob

# for drawing plt results to gui
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.tkagg as tkagg
import tkinter as Tk

from infer import _infer

#config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth = True), allow_soft_placement = True)

def preprocess(img):
    print('starting preprocess')
    cv2_img = cv2.imread(img)
    b,g,r = cv2.split(cv2_img)

    # applied clahe on 3 channel
    clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(3,3))
    contrast_b = clahe.apply(b)
    contrast_g = clahe.apply(g)
    contrast_r = clahe.apply(r)

    # merge 3 channel
    preprocessed_path = img + '_processed.JPEG'
    print("preprocessed path: " + preprocessed_path)
    contrast = cv2.merge([contrast_b,contrast_g,contrast_r])
    cv2.imwrite(preprocessed_path, contrast)
    return preprocessed_path
    
    print('end preprocess')
      
def predict(img):
    checkpoint = '/content/drive/My Drive/Testing/Weight/model-90000.pth'
    backbone = 'resnet101'
    prob_thres = 0.6
    dataset = 'voc2007'

    #for img in glob.glob('/content/drive/My Drive/Testing/both/*'):
    #for img in glob.glob('/content/drive/My Drive/Testing/both_processed/*'):

    print(img)
    _output = img + '_predicted.JPEG'
    _infer(img,_output,checkpoint,dataset,backbone,prob_thres)
    
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg
    
def prepare_figure_preprocessed(image):
    plt.figure(figsize = (5,10))
    #plt.scatter(pred_x, pred_y)
    plt.imshow(image, cmap="gray")
    return plt.gcf()
    
def launch():
    image_file = sg.PopupGetFile('Image to open')
    
    print("test")
    if image_file is None:
        print("bye")
        exit(0)

    w, h = Image.open(image_file).size
    
    # preprocess image
    preprocessed_path = preprocess(image_file)
    scaled_down_fig_image = prepare_figure_preprocessed(np.asarray(Image.open(preprocessed_path)))

    # predict
    predict(preprocessed_path)
    
    # define layout, show and read the window
    og_canvas = sg.Canvas(size=(w, h), key='og_canvas')
    preprocessed_canvas = sg.Canvas(size=(w, h), key='preprocessed_canvas')
    #output_canvas = sg.Canvas(size=(og_prediction_figure_w, og_prediction_figure_h), key='output_canvas')
    #cobb_canvas = sg.Canvas(size=(cobb_figure_w, cobb_figure_h), key='cobb_canvas')
    
    #cobb_angle_text = "The 1st cobb angle is "+ str(round(something[0][0], 2)) + " based on spine " + str(int(something[2]/2) + 1) +  " and spine " + str(int(something[3]/2) + 1) + " based on the two green lines."
    
    #cobb_angle_text2 = "The 2nd cobb angle is "+ str(round(something[0][1], 2)) + " based on spine " + str(int(something[3]/2) + 1) +  " and spine " + str(int(something[4]/2) + 1) + " based on blue and upper green lines."
    
    #cobb_angle_text3 = "The 3rd cobb angle is "+ str(round(something[0][2], 2)) + " based on spine " + str(int(something[2]/2) + 1) +  " and spine " + str(int(something[5]/2) + 1) + " based on yellow and lower green lines."
    
    layout = [[sg.Text(image_file, font='Any 18')],
              #[sg.Text(cobb_angle_text, font='Any 18')],
              #[sg.Text(cobb_angle_text2, font='Any 18')],
              #[sg.Text(cobb_angle_text3, font='Any 18')],
              [og_canvas, preprocessed_canvas], # cobb_canvas],
              #[sg.OK(pad=(((og_figure_w + preprocessed_figure_w + og_prediction_figure_w) / 2, 0), 3), size=(10, 2))]]
              [sg.OK()]]
    window = sg.Window('Cobb Angle App', return_keyboard_events=True, location=(0,0), use_default_focus=False, resizable=True).Layout(layout)
    window.Finalize()
    window.Maximize()
    og_photo = draw_figure(window.FindElement('og_canvas').TKCanvas, scaled_down_fig_image)
    #fig_photo = draw_figure(window.FindElement('preprocessed_canvas').TKCanvas, preprocessed_fig_image)
    #out_photo = draw_figure(window.FindElement('output_canvas').TKCanvas, scaled_down_prediction_image)
    #cobb_photo = draw_figure(window.FindElement('cobb_canvas').TKCanvas, cobb_fig)
    
    while True:
        event, values = window.read()
        if event == 'Maximize':
            window.Maximize()

if __name__== "__main__":
    launch()
>>>>>>> 9536051e9629c4ccec24fb110f817ff9ef08d1e9
