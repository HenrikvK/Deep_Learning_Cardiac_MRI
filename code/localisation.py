'''
Code that extracts the center of the left ventricle and cropps the picture around the center using the SAX slices, the 2CH and the 4CH views

Input: 
in \project\data\train\train\*\study should be the folders containing the ch2, ch4 and sax slice dicom data. The * represents the folder with the number of the patient
in \project\data\validate\validate\*\study should be the folders containing the ch2, ch4 and sax slice dicom data. The * represents the folder with the number of the patient

Will generate:
/data/X_train_cropped.npy -  X training data
/data/y_train_cropped.npy - labels of training data
/data/pixel_spacing_train.npy - the spacing between pixels in the picture 
/data/ids_train.npy - a vector containing which cropped image belongs to which patient
/geometry.json - extracted geometry data 
/center_points.json - coordinates of left ventricle in JSON format
/center_find/*.jpg - JPG files of heart localisation
/resized_image/*.jpg - JGP picture of cropped 64x64 picture around left ventricle
'''

import numpy as np
import os
import cv2
import re
import json
import glob
import dicom
from scipy.misc import imresize
from scipy.ndimage import imread


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom)*db + b1


def getPositionOrientationSpacingAndSizeFromGeom(geom):
    row_dircos_x = geom['ImageOrientationPatient'][0]
    row_dircos_y = geom['ImageOrientationPatient'][1]
    row_dircos_z = geom['ImageOrientationPatient'][2]

    col_dircos_x = geom['ImageOrientationPatient'][3]
    col_dircos_y = geom['ImageOrientationPatient'][4]
    col_dircos_z = geom['ImageOrientationPatient'][5]

    nrm_dircos_x = row_dircos_y * col_dircos_z - row_dircos_z * col_dircos_y
    nrm_dircos_y = row_dircos_z * col_dircos_x - row_dircos_x * col_dircos_z
    nrm_dircos_z = row_dircos_x * col_dircos_y - row_dircos_y * col_dircos_x

    pos_x = geom['ImagePositionPatient'][0]
    pos_y = geom['ImagePositionPatient'][1]
    pos_z = geom['ImagePositionPatient'][2]

    rows = geom['Rows']
    cols = geom['Columns']

    row_spacing = geom['PixelSpacing'][0]
    col_spacing = geom['PixelSpacing'][1]

    row_length = rows*row_spacing
    col_length = cols*col_spacing

    return row_dircos_x, row_dircos_y, row_dircos_z, col_dircos_x, col_dircos_y, col_dircos_z, \
        nrm_dircos_x, nrm_dircos_y, nrm_dircos_z, pos_x, pos_y, pos_z, rows, cols, \
        row_spacing, col_spacing, row_length, col_length


def rotate(dst_row_dircos_x, dst_row_dircos_y, dst_row_dircos_z,
	dst_col_dircos_x, dst_col_dircos_y, dst_col_dircos_z,
	dst_nrm_dircos_x, dst_nrm_dircos_y, dst_nrm_dircos_z,
	src_pos_x, src_pos_y, src_pos_z):
    dst_pos_x = dst_row_dircos_x * src_pos_x + dst_row_dircos_y * src_pos_y + dst_row_dircos_z * src_pos_z
    dst_pos_y = dst_col_dircos_x * src_pos_x + dst_col_dircos_y * src_pos_y + dst_col_dircos_z * src_pos_z
    dst_pos_z = dst_nrm_dircos_x * src_pos_x + dst_nrm_dircos_y * src_pos_y + dst_nrm_dircos_z * src_pos_z
    return dst_pos_x, dst_pos_y, dst_pos_z


def line_plane_intersection(point_plane_x, point_plane_y, point_plane_z,
                            point1_line_x, point1_line_y, point1_line_z,
                            point2_line_x, point2_line_y, point2_line_z,
                            plane_nrm_x, plane_nrm_y, plane_nrm_z):
    part_1_x = (point_plane_x - point1_line_x)
    part_1_y = (point_plane_y - point1_line_y)
    part_1_z = (point_plane_z - point1_line_z)
    part_2 = np.dot([part_1_x, part_1_y, part_1_z], [plane_nrm_x, plane_nrm_y, plane_nrm_z])
    line_dir_x = point2_line_x - point1_line_x
    line_dir_y = point2_line_y - point1_line_y
    line_dir_z = point2_line_z - point1_line_z

    part_3 = np.dot([line_dir_x, line_dir_y, line_dir_z], [plane_nrm_x, plane_nrm_y, plane_nrm_z])
    d_koeff = part_2/part_3
    cross_x = d_koeff*line_dir_x + point1_line_x
    cross_y = d_koeff*line_dir_y + point1_line_y
    cross_z = d_koeff*line_dir_z + point1_line_z
    return cross_x, cross_y, cross_z


def get_line_intersection(gdst, gsrc):
    dst_row_dircos_x, dst_row_dircos_y, dst_row_dircos_z, dst_col_dircos_x, dst_col_dircos_y, dst_col_dircos_z, \
        dst_nrm_dircos_x, dst_nrm_dircos_y, dst_nrm_dircos_z, dst_pos_x, dst_pos_y, dst_pos_z, dst_rows, dst_cols, \
        dst_row_spacing, dst_col_spacing, dst_row_length, dst_col_length \
        = getPositionOrientationSpacingAndSizeFromGeom(gdst)
    src_row_dircos_x, src_row_dircos_y, src_row_dircos_z, src_col_dircos_x, src_col_dircos_y, src_col_dircos_z, \
        src_nrm_dircos_x, src_nrm_dircos_y, src_nrm_dircos_z, src_pos_x, src_pos_y, src_pos_z, src_rows, src_cols, \
        src_row_spacing, src_col_spacing, src_row_length, src_col_length \
        = getPositionOrientationSpacingAndSizeFromGeom(gsrc)

    pos_x = [0, 0, 0, 0, 0, 0, 0, 0]
    pos_y = [0, 0, 0, 0, 0, 0, 0, 0]
    pos_z = [0, 0, 0, 0, 0, 0, 0, 0]

    pos_x[0] = src_pos_x
    pos_y[0] = src_pos_y
    pos_z[0] = src_pos_z

    pos_x[1] = src_pos_x + src_row_dircos_x*src_row_length
    pos_y[1] = src_pos_y + src_row_dircos_y*src_row_length
    pos_z[1] = src_pos_z + src_row_dircos_z*src_row_length

    pos_x[2] = src_pos_x + src_row_dircos_x*src_row_length + src_col_dircos_x*src_col_length
    pos_y[2] = src_pos_y + src_row_dircos_y*src_row_length + src_col_dircos_y*src_col_length
    pos_z[2] = src_pos_z + src_row_dircos_z*src_row_length + src_col_dircos_z*src_col_length

    pos_x[3] = src_pos_x + src_col_dircos_x*src_col_length
    pos_y[3] = src_pos_y + src_col_dircos_y*src_col_length
    pos_z[3] = src_pos_z + src_col_dircos_z*src_col_length

    for i in range(4):
        # Line intersection with plane
        pos_x[4+i], pos_y[4+i], pos_z[4+i] = line_plane_intersection(dst_pos_x, dst_pos_y, dst_pos_z,
                                                           pos_x[i], pos_y[i], pos_z[i],
                                                           pos_x[(i+1)%4], pos_y[(i+1)%4], pos_z[(i+1)%4],
                                                           dst_nrm_dircos_x, dst_nrm_dircos_y, dst_nrm_dircos_z)

    row_pixel = [0, 0, 0, 0, 0, 0, 0, 0]
    col_pixel = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):

        pos_x[i] -= dst_pos_x
        pos_y[i] -= dst_pos_y
        pos_z[i] -= dst_pos_z

        # rotate by the row, col and normal vectors

        pos_x[i], pos_y[i], pos_z[i] = rotate(
            dst_row_dircos_x, dst_row_dircos_y, dst_row_dircos_z,
            dst_col_dircos_x, dst_col_dircos_y, dst_col_dircos_z,
            dst_nrm_dircos_x, dst_nrm_dircos_y, dst_nrm_dircos_z,
            pos_x[i], pos_y[i], pos_z[i])


        col_pixel[i] = int(pos_x[i]/dst_col_spacing + 0.5)
        row_pixel[i] = int(pos_y[i]/dst_row_spacing + 0.5)

    # Most distant points
    xx = 4
    yy = 5
    max_dist = 0
    for i in range(4, 8):
        for j in range(i+1, 8):
            dist = (row_pixel[i] - row_pixel[j])*(row_pixel[i] - row_pixel[j]) +\
                   (col_pixel[i] - col_pixel[j])*(col_pixel[i] - col_pixel[j])
            if dist > max_dist:
                max_dist = dist
                xx = i
                yy = j

    # Return 2 most distance points of intersection plane
    return row_pixel[xx], col_pixel[xx], row_pixel[yy], col_pixel[yy]


def find_intersections_point(gsax, g2ch, g4ch):
    point_ch2_1_row, point_ch2_1_col, point_ch2_2_row, point_ch2_2_col = get_line_intersection(gsax, g2ch)
    point_ch4_1_row, point_ch4_1_col, point_ch4_2_row, point_ch4_2_col = get_line_intersection(gsax, g4ch)

    intersect = seg_intersect(np.array([point_ch2_1_row, point_ch2_1_col]),
                              np.array([point_ch2_2_row, point_ch2_2_col]),
                              np.array([point_ch4_1_row, point_ch4_1_col]),
                              np.array([point_ch4_2_row, point_ch4_2_col]))
    return intersect.tolist(), \
           point_ch2_1_row, point_ch2_1_col, point_ch2_2_row, point_ch2_2_col, \
           point_ch4_1_row, point_ch4_1_col, point_ch4_2_row, point_ch4_2_col


def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_to_grayscale_with_increase_brightness_fast(im, incr):
   min = np.amin(im.astype(float))
   max = np.amax(im.astype(float))
   out = incr*((im - min) * (255)) / (max - min)
   out[out > 255] = 255
   out = out.astype(np.uint8)

   return out


def draw_center_for_check(dcm_path, id, sax, point, points):
    debug_folder = os.path.join('..', 'calc', 'center_find')
    if not os.path.isdir(debug_folder):
        os.mkdir(debug_folder)
    ds = dicom.read_file(dcm_path)
    img = convert_to_grayscale_with_increase_brightness_fast(ds.pixel_array, 1)    
    cv2.circle(img, (int(round(point[1], 0)), int(round(point[0], 0))), 5, 255, 3)
    img = cv2.line(img, (points[1], points[0]), (points[3], points[2]), 127, thickness=2)
    img = cv2.line(img, (points[5], points[4]), (points[7], points[6]), 127, thickness=2)
    #show_image(img)
    cv2.imwrite(os.path.join(debug_folder, str(id) + '_' + sax + '.jpg'), img)

    
def resize_image(img, center):
    """
    Crop center and resize.
    :param img: image to be cropped and resized.
    """
    img_shape = (64, 64)
    centerlocal = center.copy()
    if img.shape[0] < img.shape[1]:
        img = img.T   
        #print("center before = ", centerlocal)        
        center1 = centerlocal[1]
        centerlocal[1] = centerlocal[0]
        centerlocal[0] = center1
        #print("center after = ", centerlocal)
    # we crop image from center
    short_edge = min(img.shape[:2])    
    size_square = min(100,short_edge) # determine size of square cut out of picture
    #print(size_square )
    #print("center y : ",int(round(center[1], 0)))
    yy = int((int(round(centerlocal[1], 0)) - size_square/ 2) )
    xx = int((int(round(centerlocal[0], 0)) - size_square/ 2) )
    #print("y from/to : ", yy, yy + size_square,"x from/to : ", xx, xx + size_square)
    crop_img = img[xx: xx + size_square,yy: yy + size_square ]
    img = crop_img
    img = imresize(img, img_shape)
    return img
    
def resize_image_around_center(dcm_path, id, sax, point, points):
    debug_folder = os.path.join('..', 'calc', 'resized_image')
    if not os.path.isdir(debug_folder):
        os.mkdir(debug_folder)
        
    path_all_pictures = os.path.dirname(dcm_path)
    count = 0
    
    patient_folder = os.path.join(debug_folder,str(id))
    if not os.path.isdir(patient_folder):
        os.mkdir(patient_folder)
    for pic_path in os.listdir(path_all_pictures):
        count += 1
        #print("count = ", count)
        #print(os.path.join(path_all_pictures,pic_path))
        ds = dicom.read_file(os.path.join(path_all_pictures,pic_path))    
        img = convert_to_grayscale_with_increase_brightness_fast(ds.pixel_array, 1)
        #img = cv2.line(img, (points[1], points[0]), (points[3], points[2]), 127, thickness=2)
        #img = cv2.line(img, (points[5], points[4]), (points[7], points[6]), 127, thickness=2)       
        #show_image(img)
        img = resize_image(img, point)
        
        sax_folder = os.path.join(patient_folder,str(sax))
        #show_image(img)
        if not os.path.isdir(sax_folder):
            os.mkdir(sax_folder)
        cv2.imwrite(os.path.join(sax_folder, str(count) + '.jpg'), img)

def get_centers_for_test(id, geom, debug):
    center = dict()
    ch2_el = ''
    ch4_el = ''
    for el in geom:
        matches = re.findall("(2ch_\d+)", el)
        if len(matches) > 0:
            ch2_el = el
        matches = re.findall("(4ch_\d+)", el)
        if len(matches) > 0:
            ch4_el = el
    counter = 1
    if ch2_el != '' and ch4_el != '':
        for el in geom:
            if el != ch2_el and el != ch4_el:
                print('Start extraction for test {} sax {}'.format(id, el))
                try:
                    center[el], point_ch2_1_row, point_ch2_1_col, point_ch2_2_row, point_ch2_2_col, \
                    point_ch4_1_row, point_ch4_1_col, point_ch4_2_row, point_ch4_2_col \
                        = find_intersections_point(geom[el], geom[ch2_el], geom[ch4_el])
                    #print(center[el])
                    if debug == 1:
                        draw_center_for_check(geom[el]['Path'], id, el, center[el],
                                    (point_ch2_1_row, point_ch2_1_col, point_ch2_2_row, point_ch2_2_col,
                                    point_ch4_1_row, point_ch4_1_col, point_ch4_2_row, point_ch4_2_col))
                        resize_image_around_center(geom[el]['Path'], id, el, center[el],
                                    (point_ch2_1_row, point_ch2_1_col, point_ch2_2_row, point_ch2_2_col,
                                    point_ch4_1_row, point_ch4_1_col, point_ch4_2_row, point_ch4_2_col))
                        patient_folder = os.path.join('..', 'calc', 'resized_image',str(id))
                        #print("path: ", os.path.join(patient_folder, 'Pixelspacing.npy'))
                        #print( "Pixel spacing: ",geom[el]['PixelSpacing'])
                        np.save(os.path.join(patient_folder, 'Pixelspacing.npy'), geom[el]['PixelSpacing']) 
                        #print("Pixel_spacing worked, path: ", os.path.join(patient_folder, 'Pixelspacing.npy'))    
                except:
                    print('Problem with calculation here! Number of problems: ', counter)
                    counter += 1
                    center[el] = [-1, -1]
                    


    else:
        print('Test {} miss 2ch or 4ch view of heart!'.format(id))

    return center

def read_geometry_file():
    json_path = os.path.join('..', 'calc', 'geometry.json')
    geom = dict()
    if os.path.isfile(json_path):
        f = open(json_path, 'r')
        geom = json.load(f)
        f.close()
    keys = list(geom.keys())
    for el in keys:
        geom[int(el)] = geom[el]
    for el in keys:
        geom.pop(el, None)
    return geom


def get_all_centers(start, end, debug):
    centers = dict()
    geom = read_geometry_file()
    for i in range(start, end+1):
        centers[i] = get_centers_for_test(i, geom[i], debug)
    return centers


def store_centers(centers, path):
    f = open(path, 'w')
    json.dump(centers, f)
    f.close()


def get_start_end_patients(type, input_data_path):
    split = -1
    if type == 'all':
        path = os.path.join(input_data_path, 'train\\train')
        dirs = os.listdir(path)
        max = int(dirs[0])
        for d in dirs:
            if int(d) > max:
                max = int(d)
        split = max
        path = os.path.join(input_data_path, 'validate\\validate')
        dirs += os.listdir(path)
    else:
        path = os.path.join(input_data_path, type)
        dirs = os.listdir(path)
    min = int(dirs[0])
    max = int(dirs[0])
    for d in dirs:
        if int(d) < min:
            min = int(d)
        if int(d) > max:
            max = int(d)
    return min, max, split


def find_geometry_params(start, end, split, input_data_path, output_data_path):
    if not os.path.isdir(output_data_path):
        os.mkdir(output_data_path)
    json_path = os.path.join(output_data_path, 'geometry.json')
    store = dict()
    for i in range(start, end+1):
        store[i] = dict()
        type = 'train\\train'
        if i > split:
            type = 'validate\\validate'
        path = os.path.join(input_data_path, type, str(i), 'study', '*')
        dcm_files = glob.glob(path)
        print('Total files found for test ' + str(i) + ': ' + str(len(dcm_files)))

        for d_dir in dcm_files:
            print('Read single DCMs for test' + str(i) + ': ' + d_dir)
            dfiles = os.listdir(d_dir)
            for dcm in dfiles:
                sax_name = os.path.basename(d_dir)
                dcm_path = os.path.join(d_dir, dcm)
                if (os.path.isfile(dcm_path)):
                    print('Reading file: ' + dcm_path)
                    ds = dicom.read_file(dcm_path)
                    store[i][sax_name] = dict()
                    store[i][sax_name]['ImageOrientationPatient'] = (ds.ImageOrientationPatient[0],
                                                                     ds.ImageOrientationPatient[1],
                                                                     ds.ImageOrientationPatient[2],
                                                                     ds.ImageOrientationPatient[3],
                                                                     ds.ImageOrientationPatient[4],
                                                                     ds.ImageOrientationPatient[5])
                    store[i][sax_name]['ImagePositionPatient'] = (ds.ImagePositionPatient[0],
                                                                     ds.ImagePositionPatient[1],
                                                                     ds.ImagePositionPatient[2])
                    store[i][sax_name]['PixelSpacing'] = (ds.PixelSpacing[0],
                                                          ds.PixelSpacing[1])
                    store[i][sax_name]['SliceLocation'] = (ds.SliceLocation)
                    store[i][sax_name]['SliceThickness'] = (ds.SliceThickness)
                    store[i][sax_name]['Rows'] = (ds.Rows)
                    store[i][sax_name]['Columns'] = (ds.Columns)
                    store[i][sax_name]['Path'] = dcm_path
                    break

    f = open(json_path, 'w')
    json.dump(store,f)
    f.close()

    
    
def map_studies_results():  # finds the labels (copied from data.py)
    """
    Maps studies to their respective targets.
    """
    id_to_results = dict()
    train_csv = open('data/train.csv')
    validate_csv = open('data/validate.csv')
    lines = train_csv.readlines()
    lines2 = validate_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, diastole, systole = item.replace('\n', '').split(',')
        id_to_results[id] = [float(diastole), float(systole)]
    i = 0 
    for item in lines2:
        if i == 0:
            i = 1
            continue
        id, diastole, systole = item.replace('\n', '').split(',')
        id_to_results[id] = [float(diastole), float(systole)]

    return id_to_results
    
def change_to_numpy(out_path):
    """
    Loads the cropped images data set and their labels saves it to .npy file.
    """
    print('-'*50)
    print('Writing training and validation data for heart pictures to .npy file...')
    print('-'*50)
    studies_to_results = map_studies_results()  # load the dictionary of studies to targets   
    cropped_img_path = os.path.join(out_path, 'resized_image')    
    study_ids = []  # keep a list of the patient ids    
    images = []
    image = np.zeros((30,64,64))
    images_train = []
    pixel_spacing = dict()
    for patient_path in os.listdir(cropped_img_path,):
        current_patient = int(patient_path)    

        all_sax_path = os.path.join(cropped_img_path,patient_path)
        if 'Pixelspacing.npy' in os.listdir(all_sax_path): # extract the pixel_spacing for this patient
            pixel_spacing[current_patient] = np.load(os.path.join(all_sax_path,'Pixelspacing.npy'))
            # print("pixel_spacing =" , pixel_spacing)
        else: 
            print("Warning, no pixel spacing data here!!!!!!! Patient = ",  current_patient)
        for sax_path in os.listdir(all_sax_path):
            if sax_path == 'Pixelspacing.npy':
                continue
            #print("sax_path =" , sax_path)
            study_ids.append(current_patient)
            all_time_path = os.path.join(all_sax_path,sax_path)
            t = 0
            for time_path in os.listdir(all_time_path):
                if t > 29:
                    print("Warning, this sax slice has more than 30 frames")
                    break
                time_frame_img = np.array(imread(os.path.join(all_time_path,time_path)))
                
                image[t,:,:] =  time_frame_img 
                t += 1
            if t < 30:
                print("Warning: for the sax sample exist less than 30 time frames")
                for t_ad in range(t,30):
                    image[t_ad,:,:,] = image[t_ad- t,:,:]
            images.append(image)
            image = np.zeros((30,64,64))
            
    images_train = images
    study_ids_train = study_ids
          
    X_train = []
    y_train = []
    pixel_spacing_train = []
    print("training size =", len(images_train))
    sample_nbr_train = 0
    for study_id in study_ids_train:        
        study = images_train[sample_nbr_train]
        outputs = studies_to_results[str(study_id)] 
        pixel_spac = pixel_spacing[study_id]
        X_train.append(study)
        y_train.append(outputs)
        pixel_spacing_train.append(pixel_spac)
        sample_nbr_train += 1
    X_train = np.array(X_train, dtype=np.uint8)
    y_train = np.array(y_train)
    pixel_spacing_train = np.array(pixel_spacing_train)
    study_ids_train = np.array(study_ids_train)
    
    print("X shape = ", X_train.shape)
    print("y shape = ", y_train.shape)
    print("pixel_spacing shape = ", pixel_spacing_train.shape)
    print("ids_train shape =", study_ids_train.shape)
    print("y = ", y_train)
    print("X = ", X_train)
    print("pixel_spacing_train = ", pixel_spacing_train)
    print("ids_train =", study_ids_train)
    
    np.save('data/X_train_cropped.npy', X_train)
    np.save('data/y_train_cropped.npy', y_train)
    np.save('data/pixel_spacing_train.npy', pixel_spacing_train)
    np.save('data/ids_train.npy', study_ids_train)
    
    print('Done.')
    return
   



print("Start!")
# Put train and validate folders here
input_data_path = os.path.join('..', 'project\\data')
# Results will be stored in this folder
output_data_path = os.path.join('..', 'calc')

start, end, split = get_start_end_patients('all', input_data_path)
#start = 1 
#end = 700
#split = 500
find_geometry_params(start, end, split, input_data_path, output_data_path)
centers = get_all_centers(start, end, 1)
out_path = os.path.join(output_data_path, 'center_points.json')
store_centers(centers, out_path)

# change the created cropped images to numpy vectors:
change_to_numpy(output_data_path )
print("End!")