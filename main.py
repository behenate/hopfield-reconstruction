# from os import listdir
# from os.path import isfile, join
# from hopfieldnetwork import HopfieldNetwork
# from hopfieldnetwork import images2xi, plot_network_development, DATA_DIR
# import numpy as np
# N = 150 ** 2
# hopfield_network = HopfieldNetwork(N=N)
#
#
# train_files = [f for f in listdir('chmurki') if isfile(join('chmurki', f))]
# train_paths = [join('chmurki', f) for f in train_files]
#
# test_files = [f for f in listdir('chmurki_test') if isfile(join('chmurki_test', f))]
# test_paths = [join('chmurki_test', f) for f in test_files]
#
# print("Loading images...")
# xi = images2xi(train_paths, N)
#
# print("Loading test images...")
# xi_test = images2xi(test_paths, N)
#
# print("Training network...")
# hopfield_network.train_pattern(xi)
# print("Computing energy...")
# # hopfield_network.compute_energy(xi[:, 0])
# print("Plotting network development...")
#
# einstein = np.copy(xi[:, 0])
# half_einstein = np.copy(xi[:, 0])
# half_einstein[: int(N / 5)] = -1
# hopfield_network.set_initial_neurons_state(np.copy(half_einstein))
#
# plot_network_development(
#     hopfield_network,
#     6,
#     "sync",
#     half_einstein,
#     "sync_oscillation.pdf",
#     anno_hamming=False,
# )

from hopfield_clouds import HopfieldClouds

clouds = HopfieldClouds(130**2)
a,b = clouds.get_current_image_predictions(30)
a.show()
b.show()


clouds.next_image()

clouds.next_image()
a,b = clouds.get_current_image_predictions(30)
a.show()
b.show()



clouds.next_image()
a,b = clouds.get_current_image_predictions(30)
a.show()
b.show()
