import numpy as np

def numpy_2_ply(array,output_name,threshold = 0.5):
    temp = array
    x_list = []
    y_list = []
    z_list = []

    for x in range(64):
        for y in range(64):
            for z in range(64):
                if temp[x,y,z] > threshold:
                    x_list.append(x/64)
                    y_list.append(y/64)
                    z_list.append(z/64)

    f = open(output_name,'w')
    f.write('''ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    end_header\r\n'''.format(len(x_list)))
    # 转化为ply
    for i in range(len(x_list)):
        f.write('{} {} {}\r\n'.format(x_list[i],y_list[i],z_list[i]))
    f.close()
