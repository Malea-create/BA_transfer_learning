import numpy as np
import rsatoolbox

#numpy.random.rand(10, 6) -> creates a dataset with 10 random observations of 6 channels (features)
#10 measurements were taken from 5 stimuli and which ones correspond to which stimulus
#adds a label ‘l’ vs. ‘r’ for left and right measurement channels


rsa_list = []
side = ['l', 'l', 'l', 'r', 'r', 'r'] # labels
stimulus = [0, 1, 0, 1] # Zeilen

source_labels = np.array(([8.0,7.0,6.0,8.0,7.0,6.0],[8.0,7.0,6.0,8.0,7.0,6.0],[9.0,8.0,6.0,7.0,6.0,5.0],[8.0,7.0,6.0,8.0,7.0,6.0]))
source_labels_2 = np.array(([8.0,7.0,6.0,8.0,7.0,6.0],[8.0,7.0,6.0,8.0,7.0,6.0],[9.0,8.0,6.0,7.0,6.0,5.0],[9.0,8.0,5.0,9.0,8.0,6.0]))
source_labels_3 = np.array(([1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0]))

data = rsatoolbox.data.Dataset(
    source_labels,
    channel_descriptors={'side': side},
    obs_descriptors={'stimulus': stimulus})


rdms_1 = rsatoolbox.rdm.calc_rdm(data,method='euclidean')
rsa_list.append(rdms_1)
#print("RSA: ",rdms_1)

data = rsatoolbox.data.Dataset(
    source_labels_2,
    channel_descriptors={'side': side},
    obs_descriptors={'stimulus': stimulus})


rdms_2 = rsatoolbox.rdm.calc_rdm(data,method='euclidean')
rsa_list.append(rdms_2)
#print("RSA: ",rdms_2)

data = rsatoolbox.data.Dataset(
    source_labels_3,
    channel_descriptors={'side': side},
    obs_descriptors={'stimulus': stimulus})


rdms_3 = rsatoolbox.rdm.calc_rdm(data,method='euclidean')
rsa_list.append(rdms_3)
#print("RSA: ",rdms_3)

method_list_data=["euclidean","mahalanobis","crossnobis","correlation","rho-a"]

def calculate_rdm(data):
    list = []
    for i in range(len(method_list_data)):
        result = rsatoolbox.rdm.calc_rdm(data, method = method_list_data[i])
        list.append(result[0][0])
        i += 1
    print (list)

calculate_rdm(data)

method_list_compare=["cosine","corr","corr_cov","tau-a","rho-a"]

def compare(rdm1, rdm2):
    list = []
    for i in range(len(method_list_compare)):
        result = rsatoolbox.rdm.compare(rdm1, rdm2, method = method_list_compare[i])
        list.append(result[0][0])
        i += 1
    print (list)

#compare(rdms_1,rdms_2)