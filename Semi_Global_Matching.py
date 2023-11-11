import cv2
import numpy as np
from tqdm import tqdm
import os

from Birchfield_Tomasi_dissimilarity import Birchfield_Tomasi_dissimilarity
from aggregate_cost_volume import aggregate_cost_volume


# Modify any parameters or any function itself if necessary.
# Add comments to parts related to scoring criteria to get graded.

def semi_global_matching(left_image, right_image, d,index):
    ref_is_left = (i>3)
    cost_volume, disparity = Birchfield_Tomasi_dissimilarity(left_image, right_image, d, ref_is_left)

    print("Cost volume Finished")
    np.save(os.getcwd()+"/output/Cost/cost_volume"+str(index),cost_volume)
    np.save(os.getcwd()+"/output/Intermediate_Disparity/disparity"+str(index),disparity)
    #cost_volume = np.load(os.getcwd()+"/output/Cost/cost_volume"+str(index)+".npy")
    #disparity = np.load(os.getcwd()+"/output/Intermediate_Disparity/disparity"+str(index)+".npy")
    #cost_volume과 disparity를 저장 또는 불러오는 코드
    #디버깅을 할때, 매번 새로 cost_volume을 계산하지 않아도 된다.    
    
    disparity = disparity*255/24
    disparity = disparity.astype(np.uint8)
    cv2.imwrite(os.getcwd()+"/output/Intermediate_Disparity/disparity"+str(index)+".png",disparity)
    
    aggregated_costs = aggregate_cost_volume(cost_volume)
    aggregated_disparity = aggregated_costs.argmin(axis=2)
    print("SGM Finished")
    #Cost_volume을 바탕으로 SGM을 수행, argmin을 취하여 final disparity map을 구한다.
    
    np.save(os.getcwd()+"/output/Final_Disparity/cost"+str(index),aggregated_costs)
    np.save(os.getcwd()+"/output/Final_Disparity/disparity"+str(index),aggregated_disparity)
    aggregated_disparity = aggregated_disparity*255/24
    aggregated_disparity = aggregated_disparity.astype(np.uint8)
    
    cv2.imwrite(os.getcwd()+"/output/Final_Disparity/disparity"+str(index)+".png",aggregated_disparity)
    #Disparity map과 cost를 저장하고, Disparity map을 visualize한다.

def weighted_median_filter(ref_image, image, disparity):
    #Weighted Median Filter를 수행한다.
    forward_pass = list()
    window_size = 3
    
    for i in range(window_size):
        for j in range(window_size):
            forward_pass.append((i,j))
    filtered_disparity = np.full((ref_image.shape[0],ref_image.shape[1]),0)
    
    for y in range(ref_image.shape[0]):
        for x in range(ref_image.shape[1]):
            weight = list()
            
            for idx, (dy, dx) in enumerate(forward_pass):
                if(y+dy < ref_image.shape[0] and x+dx < ref_image.shape[1] and y+dy >= 0 and x+dx >= 0):
                    weight.append([np.exp(-np.square(dx)-np.square(dy)-np.sum(np.square((ref_image[y][x]-image[y+dy][x+dx]/255))))*disparity[y+dy][x+dx],disparity[y+dy][x+dx]])
                #Disparity map에 가중치를 주어 median을 구한다. 가중치는 Spatial similarity와 color similarity를 고려한다.(Report의 2번 reference 참조)
            weight = sorted(weight, key=lambda x: x[0])
            filtered_disparity[y][x] = weight[int(len(weight)/2)][1]
            
    return filtered_disparity
            
if __name__ == "__main__":
    path = os.getcwd()+"/input/"
    img_list = os.listdir(path)
    img_list.sort()
    
    print(img_list)
    #경로를 설정하여 입력 이미지 리스트를 가져온다.
  
    d = 24
    #ref_image = cv2.imread(path+img_list[3], cv2.IMREAD_GRAYSCALE)
    ref_image = cv2.imread(path+img_list[3])
    #기준 이미지는 항상 4번 이미지이다.
    
    for i in range(len(img_list)):
        if(i==3):
            continue
        print("Matching", i ,"and", 3)
        if(i>3):
            left_image = ref_image
            #right_image = cv2.imread(path+img_list[i], cv2.IMREAD_GRAYSCALE)
            right_image = cv2.imread(path+img_list[i])
    
        else:  
            #left_image = cv2.imread(path+img_list[i], cv2.IMREAD_GRAYSCALE)
            left_image = cv2.imread(path+img_list[i])
            right_image = ref_image
        
        
        left_image = left_image.astype(np.float32)
        right_image = right_image.astype(np.float32)
        
        semi_global_matching(left_image, right_image, d,i)
        #semi_global_matching함수는 cost_volume을 구하고, SGM을 동시에 수행한다.
        #이때, RGB정보는 필요하지 않기에, gray scale, float로 변환하여 사용한다.
        #4번 이미지를 기준으로 0~2번 이미지는 왼쪽에, 5~7번 이미지는 오른쪽에 위치한다.
        #이렇게 위치정보를 전달함으로써, 하나의 이미지 pair에 대해 left_cost_volume과 right_cost_volume을 구하지 않아도 된다.
        
    ref_image = cv2.imread(path+img_list[3])
    ref_image = ref_image.astype(np.float32)
    count = np.full((ref_image.shape[0],ref_image.shape[1]),1)
    #Warping을 위해 RGB scale의 이미지를 불러온다.
    
    for i, image in enumerate(img_list):
        if(i==3):
            continue
        #왼쪽 이미지와 오른쪽 이미지의 warping 방향에 차이가 있기에 구분한다.
        if(i>3):
            right_image = cv2.imread(path+img_list[i])
            right_image = right_image.astype(np.float32)
            
            disparity = np.load(os.getcwd()+"/output/Final_Disparity/disparity"+str(i)+".npy")
            #SGM에서 구한 Final Disparity map을 불러온다.
            #disparity = weighted_median_filter(ref_image, right_image, disparity)
            #np.save(os.getcwd()+"/output/Final_Disparity/filtered_disparity"+str(i),disparity) 
            
            for y in range(right_image.shape[0]):
                for x in range(right_image.shape[1]):
                    if(x+disparity[y][x]<right_image.shape[1]):
                        for idx in range(3):
                            ref_image[y][x+disparity[y][x]][idx] = ref_image[y][x+disparity[y][x]][idx] + right_image[y][x][idx]
                        count[y][x+disparity[y][x]] = count[y][x+disparity[y][x]] + 1
            #Warping. reference image를 고정하고, 다른 이미지를 warping한다.
            #Count라는 배열을 저장하여 reference image에 해당 픽셀에 몇개의 이미지가 warping되었는지 저장한다.
            
        else:  
            left_image = cv2.imread(path+img_list[i])
            left_image = left_image.astype(np.float32)
            
            disparity = np.load(os.getcwd()+"/output/Final_Disparity/disparity"+str(i)+".npy")
            #disparity = weighted_median_filter(ref_image, left_image, disparity)
            #np.save(os.getcwd()+"/output/Final_Disparity/filtered_disparity"+str(i),disparity) 
            
            for y in range(left_image.shape[0]):
                for x in range(left_image.shape[1]):
                    if(x-disparity[y][x]>=0):
                        for idx in range(3):
                            ref_image[y][x-disparity[y][x]][idx] = ref_image[y][x-disparity[y][x]][idx] + left_image[y][x][idx]
                        count[y][x-disparity[y][x]] = count[y][x-disparity[y][x]] + 1
        
        disparity = disparity*255/24
        disparity = disparity.astype(np.uint8)
        
        cv2.imwrite(os.getcwd()+"/output/Final_Disparity/filtered_disparity"+str(i)+".png",disparity)
        
    for y in range(ref_image.shape[0]):
        for x in range(ref_image.shape[1]):
            for idx in range(3):
                ref_image[y][x][idx] = ref_image[y][x][idx] / count[y][x]
                #Count값을 이용하여 pixel-wise average를 구한다.
                
    boundary_range = d
    ground_truth = cv2.imread(os.getcwd()+"/target/gt.png")
    ground_truth = ground_truth.astype(np.float32)
    #GT 이미지를 불러온다.
    
    cropped_ground_truth = ground_truth[boundary_range:-boundary_range, boundary_range:-boundary_range]
    cropped_ref_image = ref_image[boundary_range:-boundary_range, boundary_range:-boundary_range]
    #Edge정보가 부족한 boundary를 제거한다. 
    
    print("GT size",cropped_ground_truth.shape)
    print("Estimated size",cropped_ref_image.shape)
    
    mse = 0
    for y in range(cropped_ground_truth.shape[0]):
        for x in range(cropped_ground_truth.shape[1]):
            for idx in range(3):
                mse = mse + np.sum(np.square(cropped_ground_truth[y][x][idx]-cropped_ref_image[y][x][idx]))
    mse = mse / (cropped_ground_truth.shape[0] * cropped_ground_truth.shape[1] * 3)
    #Cropped GT와 Cropped Estimated image의 MSE를 구한다.
    print("mse: {mse}".format(mse=mse))

    psnr = 10*np.log10(255**2/mse)
    #PSNR을 계산한다.
    print("psnr: {psnr}".format(psnr=psnr))