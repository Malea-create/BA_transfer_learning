from tllib.ranking import negative_conditional_entropy as nce
from tllib.ranking import log_expected_empirical_prediction as leep

import numpy as np
import rsatoolbox


source_labels = np.array([8.0,7.0,6.0,8.0,7.0,6.0,8.0,7.0,6.0,8.0,7.0,6.0]) #predicted source labels / source_labels: (N, ) elements in [0, Cs), with source class number Cs
target_labels = np.array([9.0,7.0,8.0,8.0,7.0,9.0]) #groud-truth target labels / target_labels: (N, ) elements in [0, Ct), with target class number Ct
        
result = nce(source_labels, target_labels)

print("NCE: ",result)

'''pre_trained_model_predictions = np.array([8.0,7.0,6.0,8.0,7.0,6.0,8.0,7.0,6.0,8.0,7.0,6.0]) #predictions (np.ndarray) – predictions of pre-trained model /  (N, Cs), with number of samples N and source class number Cs
target_labels = np.array([9.0,7.0,8.0,8.0,7.0,9.0]) #groud-truth labels / (N, ) elements in [0, Ct), with target class number Ct.
        
result = leep(pre_trained_model_predictions, target_labels)

print("Leep: ",result)'''

'''data = rsatoolbox.data.Dataset(np.random.rand(10, 5))
rdms = rsatoolbox.rdm.calc_rdm(data)
rsatoolbox.vis.show_rdm(rdms)'''