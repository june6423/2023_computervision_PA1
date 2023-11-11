import numpy as np

def Birchfield_Tomasi_dissimilarity(left_image, right_image, d, ref_is_left):
    # Hint: Fill undefined elements with np.inf at the end
    window_size = 7
    forward_pass = list()
    #Forward pass는 window의 좌표를 저장한다. (0,0)부터 (6,6)까지 저장한다.
    for i in range(window_size):
        for j in range(window_size):
            forward_pass.append((i,j))
            
    if(ref_is_left == False):
        left_cost_volume = np.full((left_image.shape[0],left_image.shape[1],d),np.inf)
        #reference image가 left에 있냐, right에 있냐에 따라 left_cost_volume과 right_cost_volume을 구분한다.
        #이때, cost_volume은 (height, width, disparity)의 형태를 가진다.
        
        for dis in range(d): #Disparity
            for y in range(left_image.shape[0]): #y-axis
                for x in range(left_image.shape[1]-dis): #x-axis
                        right_intensity = np.full(3,right_image[y][x+dis]) #right_intensity = lr(xr), lr+(xr), lr-(xr)
                        if(dis!=0):
                            right_intensity[2] = (right_image[y][x+dis-1]+right_image[y][x+dis])/2
                        if(x+dis+1 != left_image.shape[1]):
                            right_intensity[1] = (right_image[y][x+dis]+right_image[y][x+dis+1])/2
                            
                        left_intensity = np.full(3,left_image[y][x]) #left_intensity = ll(xl), ll+(xl), ll-(xl)
                        if(x+1 != left_image.shape[1]):
                            left_intensity[1] = (left_image[y][x+1]+left_image[y][x])/2
                        if(x != 0):
                            left_intensity[2] = (left_image[y][x-1]+left_image[y][x])/2
                    
                        dl = max([np.float32(0),left_intensity[0]-max(right_intensity),min(right_intensity)-left_intensity[0]]).astype(np.float32)
                        dr = max([np.float32(0),right_intensity[0]-max(left_intensity),min(left_intensity)-right_intensity[0]]).astype(np.float32)
                        
                        left_cost_volume[y][x][dis]=min([dl,dr])
            
        costs = np.full((left_cost_volume.shape[0],left_cost_volume.shape[1],d),np.inf)
        #Birchfield-Tomasi dissimilarity를 통해, pixel-wise similarity를 구한다.
        
        for dis in range(d):
            for y in range(left_cost_volume.shape[0]):
                for x in range(left_cost_volume.shape[1]):
                    count = 0
                    sum = np.float32(0)
                    for idx, (dy, dx) in enumerate(forward_pass):
                        if(y+dy < left_cost_volume.shape[0] and x+dx < left_cost_volume.shape[1]):
                            sum = sum + left_cost_volume[y+dy][x+dx][dis]
                            count = count + 1
                    costs[y][x][dis] = sum/count
                    #Sliding window방식으로 pixel-wise similarity의 노이즈를 줄인다.(Smoothing한다고 생각해도 무방)
                    #Edge에서 padding을 더하는 방식이 아닌, window내부 영역의 pixel-wise similarity의 평균을 구한다.
        disparity = costs.argmin(axis=2)
        #argmin을 통해 intermediate disparity를 구한다.
        
    else:    
        right_cost_volume = np.full((left_image.shape[0],left_image.shape[1],d),np.inf)
   
        for dis in range(d): #Disparity
            for y in range(left_image.shape[0]):
                for x in range(dis,left_image.shape[1]): 
                        left_intensity = np.full(3,left_image[y][x-dis]) #right_intensity = lr(xr), lr+(xr), lr-(xr)
                        if(x-dis+1 != left_image.shape[1]):
                            left_intensity[1] = (left_image[y][x-dis+1]+left_image[y][x-dis])/2
                        if(x-dis != 0):
                            left_intensity[2] = (left_image[y][x-dis-1]+left_image[y][x-dis])/2
                            
                        right_intensity = np.full(3,right_image[y][x]) #left_intensity = ll(xl), ll+(xl), ll-(xl)
                        if(x!=0):
                            right_intensity[2] = (right_image[y][x-1]+right_image[y][x])/2
                        if(x+1 != left_image.shape[1]):
                            right_intensity[1] = (right_image[y][x+1]+right_image[y][x])/2
                    
                        dl = max([np.float32(0),left_intensity[0]-max(right_intensity),min(right_intensity)-left_intensity[0]])
                        dr = max([np.float32(0),right_intensity[0]-max(left_intensity),min(left_intensity)-right_intensity[0]])
                        
                        right_cost_volume[y][x][dis]=min([dl,dr])
        
        costs = np.full((right_cost_volume.shape[0],right_cost_volume.shape[1],d),np.inf)
            
        for dis in range(d):
            for y in range(right_cost_volume.shape[0]):
                for x in range(right_cost_volume.shape[1]):
                    count = np.float32(0)
                    sum = np.float32(0)
                    for idx, (dy, dx) in enumerate(forward_pass):
                        if(y+dy < right_cost_volume.shape[0] and x+dx < right_cost_volume.shape[1]):
                            sum = sum + right_cost_volume[y+dy][x+dx][dis]
                            count = count + 1
                    costs[y][x][dis] = sum/count
        
        disparity = costs.argmin(axis=2)
    return costs, disparity
