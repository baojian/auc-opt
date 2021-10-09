import os

list_datasets = [
    'abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer', 'cardio_3',
    'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german', 'ionosphere',
    'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
    'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks', 'pen_digits_0', 'pen_digits_5', 'pima',
    'satimage_4', 'scene', 'seismic', 'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf',
    'spectrometer', 'splice', 'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
    'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt', 'yeast_me1', 'yeast_me2', 'yeast_ml8']
print(len(list_datasets))
root = '/home/baojian/Dropbox/pub/2020/NIPS-2020/supp-and-code/datasets/'
root1 = '/home/baojian/Dropbox/pub/2020/NIPS-2020/supp-and-code/results/'
for dataset in list_datasets:
    os.system('mv ' + root + r't_sne_2d_%s.pkl' % dataset +
              ' ' + root + r'%s/t_sne_2d_%s.pkl' % (dataset, dataset))
