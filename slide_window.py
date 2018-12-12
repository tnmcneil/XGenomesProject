#! /usr/bin/env python3
import numpy as np
import cv2
import sys
import time
import os
import shutil

# Copyright 2018 Yang Kaiyu kyyang@bu.edu
def rotatecol(image, angle=3, flip=0):
    """A better rotate method"""
    size_y, size_x = image.shape
    size_x //= 2
    size_y //= 2
    M = cv2.getRotationMatrix2D((size_x,size_y),angle+180*flip,1)
    nY, nX = np.uint32(image.shape @ np.abs(M[:,:2]))
    M[0, 2] += (nX / 2) - size_x
    M[1, 2] += (nY / 2) - size_y
    return cv2.warpAffine(image,M,(nX,nY))
    
def mk_Dir(name):
    """prepare output dir"""
    if name not in os.listdir('./'):
        os.mkdir('./'+name)
    else:
        shutil.rmtree('./'+name)
        os.mkdir('./'+name)

def gen_img(input_folder, output_folder, win_size=[200,100], step=[200,100], ratio=1, rotation=[0]):
    """generate images"""
    ori_img = [["./"+input_folder+"/"+aname,aname] for aname in os.listdir("./"+input_folder+"/")]
    
    i = 0
    for a_img in ori_img:
        image_file = cv2.imread(a_img[0],0)
        image_file = rotatecol(image_file,rotation[i],0) # correct orientation
        i += 1
        size_y, size_x = image_file.shape
        
        x_count = int((size_x - win_size[1])/step[1])
        y_count = int((size_y - win_size[0])/step[0])
        
        totalnumber = x_count*y_count
        nameformat = '0'+str(len(str(totalnumber)))+'d'
        count = 0
        for y_co in range(y_count):
            for x_co in range(x_count):
                sub_img = image_file[y_co*step[0]:y_co*step[0]+win_size[0],x_co*step[1]:x_co*step[1]+win_size[1]]
                sub_img = cv2.resize(sub_img,None,fx=ratio,fy=ratio)
                cv2.imwrite("./"+output_folder+"/"+os.path.splitext(a_img[1])[0]+"_"+format(count,nameformat)+".jpg",sub_img)
                count +=1
                print("Progress",a_img[1],str(int(count/totalnumber*100))+"%", end="\r")
        

def main():
    ############################################################
    ############################################################
    ClassA_ori_folder = "Lambda_fake"  # Input folder
    ClassB_ori_folder = "T7_fake"  # Input folder
    
    ClassA_target_folder = "Lambda_gen"  # Output folder
    ClassB_target_folder = "T7_gen"  # Output folder
    
    ClassA_rotation = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # Manually correct the image's orientation (in degree)
    ClassB_rotation = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # The length of this list should match the number of imput images.
    
    Window_size = [1280,320]  # Size of cutting window [height, width]
    Step_size = [640,160]  # Step size for the moving window [vertical, horizontal]
    Output_ratio = 1/2  # output scale factor
    ############################################################
    ############################################################
    
    mk_Dir(ClassA_target_folder)
    mk_Dir(ClassB_target_folder)
    
    print("Generate:",ClassA_ori_folder)
    gen_img(ClassA_ori_folder, ClassA_target_folder, Window_size, Step_size, Output_ratio, ClassA_rotation)
    print("Generate:",ClassB_ori_folder)
    gen_img(ClassB_ori_folder, ClassB_target_folder, Window_size, Step_size, Output_ratio, ClassB_rotation)
    
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

    