How to execute : python Semi_Global_Matching.py

Result location
    Default Noise level : 25(noise25.png)

    cost_volume : ./output/Cost/cost_volume.npy
    intermediate disparity : ./output/Intermediate_Dispairty/disparity.npy (Or png for visualization)
    Disparity after SGM : ./output/Final_Disparity/disparity.npy (Or png for visualization)
    Disparity after Weighted median filter : ./output/Final_Disparity/filtered_disparity.npy (Or png for visualization)
    Aggregated_cost volume : ./output/Final_Disparity/cost.npy
    Estimated_image : ./output/estimated.png
    
Version of my environment list(conda list)

_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
ca-certificates           2023.08.22           h06a4308_0  
certifi                   2023.7.22                pypi_0    pypi
idna                      3.4                      pypi_0    pypi
ld_impl_linux-64          2.38                 h1181459_1  
libffi                    3.4.4                h6a678d5_0  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libstdcxx-ng              11.2.0               h1234567_1  
ncurses                   6.4                  h6a678d5_0  
opencv-python             4.8.0.76                 pypi_0    pypi
openssl                   3.0.10               h7f8727e_2  
pillow                    10.0.1                   pypi_0    pypi
pip                       23.2.1           py38h06a4308_0  
python                    3.8.18               h955ad1f_0  
readline                  8.2                  h5eee18b_0  
requests                  2.31.0                   pypi_0    pypi
rich                      13.5.3                   pypi_0    pypi
setuptools                68.0.0           py38h06a4308_0  
sqlite                    3.41.2               h5eee18b_0  
tk                        8.6.12               h1ccaba5_0  
torch                     2.0.1                    pypi_0    pypi
torchvision               0.15.2                   pypi_0    pypi
tqdm                      4.66.1                   pypi_0    pypi
urllib3                   2.0.4                    pypi_0    pypi
wheel                     0.38.4           py38h06a4308_0  
xz                        5.4.2                h5eee18b_0  
zlib                      1.2.13               h5eee18b_0  