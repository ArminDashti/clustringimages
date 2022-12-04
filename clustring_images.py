import tensorflow as tf
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import shutil
import glob
from sklearn.cluster import AffinityPropagation

class image_clustring:
    def __init__(self, input_path, output_path, file_type=".jpg"):
        self.input_path = input_path
        self.output_path = output_path
        self.file_type = file_type
        self.list_imgs_path_input = []
        self.model = None
        self.extracted_features =None
        self.dict_imgs_extracted_features_global = {}
        self.imgs_features_numpy = None       
        self.clustred_images = None
        self.dim_reduction_imgs = None
        self.pca = None
        self.ap_clustering = None
        self.cluster_number = None
        self.run()
    
    def input_folder_path(self):
        list_imgs = []
        with os.scandir(self.input_path) as files:
            for file in files:
                if file.name.endswith(self.file_type):
                    list_imgs.append(self.input_path+file.name)
        return list_imgs
                    
    def func_download_models_fit(self):
        model = tf.keras.applications.VGG19()
        model = tf.keras.applications.ResNet50()
        self.model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    def extract_img_features(self, img):
        loaded_image = tf.keras.utils.load_img(img, target_size=(224,224))
        numpy_image = np.array(loaded_image)
        numpy_image = numpy_image.reshape(1,224,224,3)
        preprocessed_image = preprocess_input(numpy_image)
        return self.model.predict(preprocessed_image, use_multiprocessing=True)
        
    def extract_imgs_features(self, list_imgs):
        dict_imgs_extracted_features = {}
        for img in list_imgs:
            extract_img = self.extract_img_features(img)
            dict_imgs_extracted_features[img] = extract_img
            
        imgs_features_numpy = np.array(list(dict_imgs_extracted_features.values()))
        return imgs_features_numpy.reshape(-1,2048)
            
    def dim_reduction(self, imgs_features, n_components_p=None):
        self.pca = PCA(n_components=5, random_state=22)
        self.pca.fit(imgs_features)
        return self.pca.transform(imgs_features)
        
    def cluster_imgs(self, imgs_list):
        self.ap_clustering = AffinityPropagation().fit(imgs_list)
        pred = self.ap_clustering.predict(imgs_list)
        self.cluster_number = np.unique(pred)
        print("Get " + str(self.cluster_number) + " Cluster")
        
    def create_folder_output(self):
        for i in self.cluster_number:
            path = os.path.join(self.output_path, str(i))
            os.mkdir(path)
        
    def output(self):
        print('======== START Clustring')
        self.create_folder_output()
        list_imgs_path = self.input_folder_path()
        i = 0
        for img in list_imgs_path:
            image_cluster = self.predict_img(img)[0]
            shutil.copyfile(img, self.output_path + str(image_cluster) + "/" + str(i) + self.file_type)
            i = i + 1
        
    def predict_img(self, img):
        extract_img_features = self.extract_img_features(img)
        pca = self.pca.transform(extract_img_features)
        predicted = self.ap_clustering.predict(pca)
        return predicted
            
    def run(self):
        list_imgs_path = self.input_folder_path()
        self.func_download_models_fit()
        extract_imgs_features = self.extract_imgs_features(list_imgs_path)
        dim_reduction_imgs = self.dim_reduction(extract_imgs_features)
        self.cluster_imgs(dim_reduction_imgs)
        self.output()