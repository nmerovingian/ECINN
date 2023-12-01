import os 
import glob
import re
import shutil
import numpy as np
from scipy import interpolate
from os import listdir


result = glob.glob('*.{}'.format('csv'))


def toDimensionalPotential(x):
    return x / (96485/(8.314*298))
def toDimensionlessPotential(x):
    return x * (96485/(8.314*298))


def flux_sampling(time_array,df_FD,maxT):
    interpolated_flux = np.zeros_like(time_array)
    time_array_length = len(time_array)
    df_FD_forward = df_FD.iloc[:int(len(df_FD)/2)]
    df_FD_reverse = df_FD.iloc[int(len(df_FD)/2):]
    forward_scan_time_array = time_array[:int(time_array_length/2)]
    reverse_scan_time_array  = time_array[int(time_array_length/2):]
    f_forward = interpolate.interp1d(np.linspace(0,maxT/2.0,num=len(df_FD_forward)),df_FD_forward.iloc[:,1])
    f_reverse = interpolate.interp1d(np.linspace(maxT/2.0,maxT,num=len(df_FD_reverse)),df_FD_reverse.iloc[:,1])
    interpolated_flux[:int(time_array_length/2)] = f_forward(forward_scan_time_array)
    interpolated_flux[int(time_array_length/2):] = f_reverse(reverse_scan_time_array)
    return interpolated_flux



def exp_flux_sampling(time_array,df_exp,FullScanT,PortionAnalyzed=0.75):
    interpolated_flux = np.zeros_like(time_array)
    time_array_length = len(time_array)
    df_exp_forward = df_exp.iloc[:int(len(df_exp)*0.5)]
    df_exp_reverse = df_exp.iloc[int(len(df_exp)*0.5):int(len(df_exp)*PortionAnalyzed)]
    forward_scan_time_array = time_array[time_array<FullScanT*0.5]
    reverse_scan_time_array  = time_array[time_array>=FullScanT*0.5]
    assert time_array_length == len(forward_scan_time_array) + len(reverse_scan_time_array)
    f_forward = interpolate.interp1d(np.linspace(0,FullScanT*0.5,num=len(df_exp_forward)),df_exp_forward.iloc[:,1])
    f_reverse = interpolate.interp1d(np.linspace(FullScanT*0.5,FullScanT*PortionAnalyzed,num=len(df_exp_reverse)),df_exp_reverse.iloc[:,1])
    interpolated_flux[:len(forward_scan_time_array)] = f_forward(forward_scan_time_array)
    interpolated_flux[len(forward_scan_time_array):] = f_reverse(reverse_scan_time_array)
    return interpolated_flux




dir = os.getcwd()
def find_csv(path_to_dir=None,suffix='.txt'):    #By default, find all csv in current working directory 
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) and 'Experimental' not in filename and 'One Electron Reduction' not in filename]

def find_experimental_csv(path_to_dir=None,preffix = 'Experimental',suffix='.csv'):  # By default, find all csv starts with experimenatl 
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) and 'Experimental' in filename]


def expParameters(file_name):
    file_name = file_name.replace('.csv','')
    pattern = re.compile(r'sigma=([\d.]+[eE][+-][\d]+)')
    sigma = float(pattern.findall(file_name)[0])

    pattern = re.compile(r'theta_i=([\d.]+[eE][+-][\d]+)')
    theta_i = float(pattern.findall(file_name)[0])

    pattern = re.compile(r'theta_v=([+-][\d.]+[eE][+-][\d]+)')
    theta_v = float(pattern.findall(file_name)[0])

    pattern = re.compile(r'dA=([\d.]+[eE][+-][\d]+)')
    dA = float(pattern.findall(file_name)[0])

    pattern = re.compile(r'dB=([\d.]+[eE][+-][\d]+)')
    dB = float(pattern.findall(file_name)[0])


    return sigma, theta_i,theta_v,dA,dB




def find_sigma(CV):
    CV = CV.replace('.csv','')
    pattern = re.compile(r'var1=([\d.]+[eE][+-][\d]+)')

    var1 = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var2=([\d.]+[eE][+-][\d]+)')

    var2 = float(pattern.findall(CV)[0])


    pattern = re.compile(r'var3=([\d.]+[eE][+-][\d]+)')

    var3 = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var4=([\d.]+[eE][+-][\d]+)')
    
    var4 = float(pattern.findall(CV)[0])


    pattern = re.compile(r'var5=([\d.]+[eE][+-][\d]+)')
    
    var5 = float(pattern.findall(CV)[0])
    
    pattern = re.compile(r'var6=([\d.]+[eE][+-][\d]+)')
    
    var6 = float(pattern.findall(CV)[0])
    return var1,var2,var3,var4,var5,var6



def find_conc(CV):
    CV = CV.replace('.csv','')
    pattern = re.compile(r'Point=([A-Z])')

    point = pattern.findall(CV)[0]

    pattern = re.compile(r'Theta=([-]?[\d.]+[eE][+-][\d]+)')
    
    Theta = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var1=([\d.]+[eE][+-][\d]+)')

    var1 = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var2=([\d.]+[eE][+-][\d]+)')

    var2 = float(pattern.findall(CV)[0])


    pattern = re.compile(r'var3=([\d.]+[eE][+-][\d]+)')

    var3 = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var4=([\d.]+[eE][+-][\d]+)')
    
    var4 = float(pattern.findall(CV)[0])


    pattern = re.compile(r'var5=([\d.]+[eE][+-][\d]+)')
    
    var5 = float(pattern.findall(CV)[0])

    pattern = re.compile(r'var6=([\d.]+[eE][+-][\d]+)')
    
    var6 = float(pattern.findall(CV)[0])

    return point, Theta, var1,var2,var3,var4,var5,var6


def find_point(CVs,point):
    pattern=re.compile(f'Point={point}.*')

    match = []

    for CV in CVs:
        m = pattern.findall(CV)
        if m is not None and len(m) > 0:
            match.append(m[0])

    return match

def move_files():
    files = find_csv()
    for file in files:
        if 'radial' in file:
            shutil.move(f'./{file}',f'./radial/{file}')
        elif 'linear' in file:
            shutil.move(f'./{file}',f'./linear/{file}')


if __name__ == "__main__":
    move_files()



def format_func_dimensionla_potential(value,tick_number):
    #convert dimensionless potential to dimensional potential in mV
    value = value / 96485 * 8.314*298.0 *1e3
    return (f'{value:.2f}')