import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import ImageDatasetFromHDF5


photon_file_path = '/scratch/gsoc/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5'
electron_file_path = '/scratch/gsoc/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5'

photon_dset = ImageDatasetFromHDF5(photon_file_path)
electron_dset = ImageDatasetFromHDF5(electron_file_path)

# photon = photon_dset[0][0]
# photon_label = photon_dset[0][1]
# electron = electron_dset[0][0]
# electron_label = electron_dset[0][1]

# print(photon_label, electron_label)

# print(photon.shape)

# print(photon_dset[0])
# print(electron_dset[1][0].shape)

# plt.imshow(photon[:, :, 0], cmap='gray')
# plt.savefig("photon_E.png")

# plt.imshow(photon[:, :, 1], cmap='gray')
# plt.savefig("photon_t.png")


# plt.imshow(electron[:, :, 0], cmap='gray')
# plt.savefig("electron_E.png")

# plt.imshow(electron[:, :, 1], cmap='gray')
# plt.savefig("electron_t.png")