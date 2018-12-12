#! /usr/bin/env python3
import numpy as np
import cv2
import sys
import time

# Copyright 2018 Yang Kaiyu kyyang@bu.edu
def pasteimg(target, obj, offset=[0,0]):
    """General image overlay with overflow prevention"""
    offset[0],offset[1] = int(offset[0]+0.5),int(offset[1]+0.5)
    temp = 255 - target[offset[1]:offset[1]+obj.shape[0],offset[0]:offset[0]+obj.shape[1]]
    target[offset[1]:offset[1]+obj.shape[0],offset[0]:offset[0]+obj.shape[1]] += np.minimum(temp,obj)

def pasteimg32(target, obj, offset=[0,0]):
    """High speed image overlay without overflow prevention. USE uint32!!"""
    offset[0],offset[1] = int(offset[0]+0.5),int(offset[1]+0.5)
    target[offset[1]:offset[1]+obj.shape[0],offset[0]:offset[0]+obj.shape[1]] += obj

def genspot(size=10, blur=5, alpha=1):
    """Generate a spot"""
    size, blur = int(size), int(blur)
    spot_base = np.zeros([size*2,size*2],dtype=np.uint8)
    if not blur%2:
        blur -= 1
    cv2.circle(spot_base,(size,size),max(size-(blur+1)//2,1),255*alpha,thickness=-1)
    return cv2.GaussianBlur(spot_base,(blur,blur),0)

def randspotlib(libsize, _size=10, _blur=5, _alpha=1, randomness=[1,0]):
    """Pre generate a random spot library"""
    randomlist = np.random.rand(int(libsize),2)*randomness
    output = []
    for seed in randomlist:
        output.append(genspot(_size,_blur*(1+seed[0]*0.75),_alpha*(1+0.5*randomness[1]-seed[1])))
    return output

def seqremap(seq, size=512, noise=[1,1]):
    """Remap DNA sequence to graph coordinates with adjustable noise"""
    output = np.random.rand(len(seq),2)*noise
    output[:,1] += np.array(seq)/np.max(seq)*size
    return np.round(output)

def drawseq(seq, draw_lib, seq_size=400, seq_noise=[1,1], frag=[0,1]):
    """Draw a single DNA molecule,using the spot library"""
    spot_size = draw_lib[0].shape[0]
    seq_size = int(seq_size)
    temp_img = np.zeros([seq_size+spot_size+max(seq_noise[1],1),spot_size+max(seq_noise[0],1)],dtype=np.uint32)
    po = seqremap(seq,seq_size,seq_noise)
    selspot = np.uint32(np.random.rand(len(po))*len(draw_lib))
    frag = [temp_img.shape[0]*frag[0],temp_img.shape[0]*frag[1]]
    for j in range(len(po)):
        if frag[0]<= po[j][1] <=frag[1]:
            pasteimg32(temp_img,draw_lib[selspot[j]],po[j].tolist())
    return np.uint8(np.minimum(temp_img[int(frag[0]):int(frag[1]+spot_size+max(seq_noise[1],1)),:],255))

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

def randomXY(size, XYdens=[25,3],XYrand=[1,1]):
    """Generate an evenly distributed random XY map"""
    size_y, size_x = size
    x_gap = size_x//XYdens[0]
    y_gap = size_y//XYdens[1]
    XYmap = []
    for i in range(int(XYdens[1])):
        for j in range(int(XYdens[0])):
            XYmap.append([j*x_gap,i*y_gap])
    
    rand = np.random.rand(int(XYdens[0])*int(XYdens[1]),2)*XYrand*[x_gap,y_gap]
    XYmap = XYmap + rand
    return XYmap.tolist()

def main():
    # Define Binding Behavior
    Lambda_seq = [449, 1099, 1231, 1354, 1463, 1557, 1817, 2004, 2108, 2494, 2548, 3048, 3054, 3115, 3651, 3675, 4827, 5147, 5434, 6163, 6283, 6332, 6436, 6707, 6869, 7153, 7290, 7495, 7902, 8300, 8437, 9205, 9244, 9689, 9866, 9889, 10149, 10175, 10473, 10745, 10924, 10930, 10945, 11167, 11271, 11860, 12099, 12127, 12162, 12176, 12214, 12266, 12344, 12358, 12473, 12538, 12570, 12662, 12665, 12680, 12839, 13334, 13700, 13831, 13905, 14091, 14158, 14182, 14322, 14523, 14567, 14798, 14813, 15024, 15072, 15105, 15224, 16452, 16647, 16690, 16921, 16991, 17206, 17360, 17417, 17552, 17697, 17829, 18030, 18201, 18254, 18526, 18601, 18714, 19330, 19458, 19534, 19541, 19987, 20114, 20306, 20348, 20791, 20865, 21139, 21403, 22115, 22644, 22715, 23005, 23334, 24315, 25827, 27014, 27360, 28084, 29044, 29116, 30046, 30499, 31041, 31355, 31441, 31764, 32716, 32949, 33276, 33285, 33728, 34721, 35071, 35102, 35600, 35624, 35905, 36177, 36616, 37694, 38004, 38568, 38775, 39174, 39641, 39786, 40228, 40436, 40520, 40545, 41012, 41766, 42091, 42951, 43177, 43277, 43511, 43786, 43920, 44004, 45221, 45271, 45296, 45341, 45536, 45798, 46055, 46102, 46578]
    T7_seq = [1865, 2146, 2399, 2689, 2879, 3017, 3380, 3761, 3864, 3872, 4051, 4184, 4434, 4507, 4771, 5826, 7314, 7407, 7706, 8209, 9217, 9709, 10808, 10998, 11082, 13031, 13044, 13461, 13875, 15717, 17296, 17771, 18289, 19627, 19777, 19816, 20272, 21985, 22069, 23040, 23062, 23114, 23355, 24080, 24289, 24932, 26587, 27124, 27511, 27763, 28742, 29017, 29599, 30888, 31215, 31785, 31917, 32358, 32575, 33030, 33120, 33488, 34392, 35218, 36057, 36691, 37404, 37428, 37506]
    
    ######################################################################
    ######################################################################
    # Define Drawing parameters
    Output_number = 100 # How many output images (int) (0<)
    Output_size = [2560,2560] # Output image size [height, width] (int) (0<)
    Target_sequence = T7_seq # Which sequence to draw (list of int numbers) (0<=)
    
    out_box = 1 # flag for output annotation in .csv (0 or 1)
    box_tag = "T7" # tag for output annotation in .csv (string)
    
    Binding_size = 5 # Binding Spot Radius (int or float) (0<)
    Binding_soft = 5 # Binding Spot Fuzziness (int or float) (0< & <= Binding_size)
    Binding_str = 1 # Light intensity of the binding point (float) (0~1)
    
    Density = [20,4] # Molecule density [x,y] (int) (0<)  35,6
    
    Sequence_rand = [1,2] # Randomness of each binding location [x,y] (int or float) (0<=)
    Binding_rand = [1,0.5] # Randomness of each binding point [soft,str] (int or float) (0<=)
    Angle_rand = [1,1,1] # Randomness of alignment [noise,max,bias] (int or float) (0<=)
    flip_flag = 1 # allow random flip 0=no, 1=allow 
    frag_rand = 0.7 # possibility of break (float) (0~1)
    length_rand = 0.2 # Randomness of molecule length (float) (0~1)
    Position_rand = [0.4,1] # Randomness of molecule position[x,y] (float) (0~1)
    
    Molecule_size = 800 # Drawing length of each DNA molecule (int) (0< & <image height)
    Molecule_counts = Density[0]*Density[1] # How many molecules on one photo (don't change)
    
    global_noise = 0.01 # Add possibility of global_noise (float) (0~1)
    ######################################################################
    ######################################################################
    
    # Setup environment
    nameformat = '0'+str(len(str(Output_number)))+'d' # generate file name format
    Buffer_size = [int(Output_size[0]+Molecule_size*2),int(Output_size[1]+Molecule_size)] # [height, width]
    if out_box:
        coord_file = open("annotation.csv",'w')
    Output_yx = (Molecule_size,Molecule_size//2,Molecule_size+Output_size[0],Molecule_size//2+Output_size[1])
    
    now = time.time() # Start timer
    # Start generating
    for count in range(Output_number):
        out_img = np.zeros(Buffer_size,dtype=np.uint8) # image buffer
        
        rand_lib = np.random.rand(Molecule_counts,7) # generate random matrix
        rand_lib[:,0] = (rand_lib[:,0]-0.5)*Angle_rand[0] # random list for angle noise
        rand_lib[:,1] = np.uint8((rand_lib[:,1]+0.5)*flip_flag) # random list for flip
        rand_lib[:,2] = (rand_lib[:,2]+0.5)*length_rand+1 # random list for molecule size
        
        rand_lib[:,3:5] = np.uint8(rand_lib[:,3:5]+frag_rand) # random list for break point
        rand_lib[:,4] = 1 - rand_lib[:,4]
        
        degree_buff = (np.random.rand()-0.5)*2*Angle_rand[1]
        degree_bias = (np.random.rand()-0.5)*2*Angle_rand[2]
        
        rand_map = randomXY(Buffer_size, Density, Position_rand)
        
        coord_buffer = ""
        for i in range(Molecule_counts):
            rand_lib_size = int(max((Binding_rand[0]+Binding_rand[1])*7,1)) # spot library
            random_spots = randspotlib(rand_lib_size, Binding_size, Binding_soft, Binding_str, Binding_rand) # spot library
            
            first_b, second_b = min(rand_lib[i,5],rand_lib[i,6]), max(rand_lib[i,5],rand_lib[i,6]) # break molecule
            fragment = [rand_lib[i,3]*first_b, rand_lib[i,4] + second_b]
            a_DNA = drawseq(Target_sequence, random_spots, Molecule_size*rand_lib[i,2], Sequence_rand, fragment) # generate single molecule
            
            degree_buff += rand_lib[i,0] # random alignment
            degree_buff = min(degree_buff,Angle_rand[1])
            degree_buff = max(degree_buff,-Angle_rand[1])
            a_DNA = rotatecol(a_DNA,degree_buff+degree_bias,rand_lib[i,1]) # add rotation

            try:
                pasteimg(out_img,a_DNA,rand_map[i]) # paste molecule at a random location
                if out_box: # output annotation data
                    box_ul_x = min(max(int(rand_map[i][0] - Output_yx[1]),0),Output_size[1]-1)
                    box_ul_y = min(max(int(rand_map[i][1] - Output_yx[0]),0),Output_size[0]-1)
                    box_dr_x = min(max(int(rand_map[i][0] + a_DNA.shape[1] - Output_yx[1]),0),Output_size[1]-1)
                    box_dr_y = min(max(int(rand_map[i][1] + a_DNA.shape[0] - Output_yx[0]),0),Output_size[0]-1)
                    if(box_ul_x!=box_dr_x)and(box_ul_y!=box_dr_y): # check for clipping
                        coord_buffer += format(count,nameformat)+".jpg," +str(box_ul_x)+","+str(box_ul_y)+","+str(box_dr_x)+","+str(box_dr_y)+","+box_tag+"\n"
            except:
                pass
        if out_box:
            coord_file.write(coord_buffer)
        out_img = out_img[Output_yx[0]:Output_yx[2],Output_yx[1]:Output_yx[3]]
        # Add global noise
        rand_map = randomXY(out_img.shape, [out_img.shape[1]*global_noise+1,out_img.shape[0]*global_noise+1], [1,1])
        rand_lib_size = int(max((Binding_rand[0]+Binding_rand[1])*7,1)) # spot library
        random_spots = randspotlib(rand_lib_size, Binding_size, Binding_soft, Binding_str, Binding_rand) # spot library
        rand_lib = (np.random.rand(len(rand_map))*rand_lib_size).astype(int)
        i = 0
        for cord in rand_map:
            try:
                pasteimg(out_img, random_spots[rand_lib[i]], offset=cord)
            except:
                pass
            i += 1
        
        cv2.imwrite(box_tag+format(count,nameformat)+".jpg",np.uint8(np.minimum(out_img,255))) # save file
        print("Progress",str(int(count/Output_number*100))+"%", end="\r")
    
    print("Finish in:",time.time()-now,"sec")
    coord_file.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
