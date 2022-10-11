import h5py
import numpy as np
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, CarliniLInfMethod
from art.estimators.classification import KerasClassifier
from tensorflow import keras
import tensorflow as tf
import os

from cem_helper_func import prepare_data

tf.compat.v1.disable_eager_execution()

epsilon = np.arange(0.01, 0.161, 0.015)
# epsilon = [0.025]

(x_train, y_train), (x_test, y_test) = prepare_data(subtract_pixel_mean=True, shuffle=True, seed=101)

# Load models
path_source = os.path.join(os.getcwd(), 'saved_models/cifar10/source/0/cifar10_ResNet20v1_model.h5')
model_source = keras.models.load_model(path_source)

wrapped_classifier = KerasClassifier(model=model_source)

for e in epsilon:
    print(f"Epsilon: {e}")

    bim = BasicIterativeMethod(estimator=wrapped_classifier, eps=e, eps_step=0.01, max_iter=100, targeted=False)
    pgd = ProjectedGradientDescent(estimator=wrapped_classifier, norm=2, eps=e, eps_step=0.01, max_iter=10, targeted=False)
    fgm = FastGradientMethod(estimator=wrapped_classifier, norm=np.inf, eps=e, eps_step=e, targeted=False)
    cwl = CarliniLInfMethod(classifier=wrapped_classifier, targeted=False, confidence=0.5, learning_rate=0.01, max_iter=50, eps=e)

    print("Generate BIM")
    bim_examples = bim.generate(x_test[:100])
    print("Generate PGD")
    pgd_examples = pgd.generate(x_test[:100])
    print("Generate FGM")
    fgm_examples = fgm.generate(x_test[:100])
    print("Generate CWL")
    cwl_examples = cwl.generate(x_test[:100])

    dataset_path = f'adversarial_examples/e_{e}/'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    with h5py.File(f'{dataset_path}/adversarial_examples.h5', 'w') as h5f:
        h5f.create_dataset('BIM', data=bim_examples)
        h5f.create_dataset('PGD', data=pgd_examples)
        h5f.create_dataset('FGM', data=fgm_examples)
        h5f.create_dataset('CWL', data=cwl_examples)
