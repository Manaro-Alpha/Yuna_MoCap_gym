import numpy as np

motion_array = np.load('/home/mananaro/Yuna_MoCap_gym/Yuna_train_data.npy')
# motion_array_pos = np.zeros(motion_array.shape)
# for i in range(motion_array.shape[0]):
#     for j in range(motion_array.shape[1]):
#         motion_array_pos[i][j] = motion_array[i][j][1]

print(motion_array.shape)