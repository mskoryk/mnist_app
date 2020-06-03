import streamlit as st
import os
from mnist_predict import *

import cv2

def file_selector(folder_path='.'):
    filenames = [ f for f in os.listdir(folder_path) if f.lower().endswith('png')]
    filenames = [''] + filenames
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


# Text/Title
st.title("Digit recognizer")


# filename = st.file_uploader("Choose file...")


filename = file_selector()
if len(filename.split('/')[-1]) > 0:
    st.write('You selected {}'.format(filename))
    st.image(filename, format='PNG', width=500, caption='Handwritten digit')
    image = cv2.imread(filename, 0)    
    st.write(mnist_predict(image))

