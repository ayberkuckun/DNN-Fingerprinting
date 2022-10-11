import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from cem_helper_func import prepare_data


surrogate_no = reference_no = 5

epsilon = np.arange(0.01, 0.161, 0.015)
# epsilon = [0.025]

(_, _), (_, y_test) = prepare_data(subtract_pixel_mean=True, shuffle=True, seed=101)

# Load models
path_source = os.path.join(os.getcwd(), 'saved_models/source/0/ResNet20v1_model.h5')
model_source = keras.models.load_model(path_source)

models_surrogate = []
for i in range(surrogate_no):
    path_surrogate = os.path.join(os.getcwd(), f'saved_models/cifar10/surrogate/{i+8}/ResNet20v1_model.h5')
    model_surrogate = keras.models.load_model(path_surrogate)
    models_surrogate.append(model_surrogate)

models_reference = []
for i in range(reference_no):
    path_reference = os.path.join(os.getcwd(), f'saved_models/cifar10/reference/{i+8}/ResNet20v1_model.h5')
    model_reference = keras.models.load_model(path_reference)
    models_reference.append(model_reference)

total_scores = []
for e in epsilon:
    print(f"Epsilon: {e}")
    print("Load Adversarial Examples")
    with h5py.File(f'adversarial_examples/e_{e}/adversarial_examples.h5', 'r') as h5file1:
        bim_examples = np.array(h5file1["BIM"])
        pgd_examples = np.array(h5file1["PGD"])
        fgm_examples = np.array(h5file1["FGM"])
        cwl_examples = np.array(h5file1["CWL"])

    print("Load CEM Examples")
    with h5py.File(f'fingerprints/e_{e}/01_29/fingerprint.h5', 'r') as h5file2:
        fingerprint = np.array(h5file2["fingerprint"])

    cem_examples = np.reshape(np.array(fingerprint), (-1, 32, 32, 3))

    adversarial_examples = [bim_examples, pgd_examples, fgm_examples, cwl_examples, cem_examples]

    scores_for_e = []
    for examples_unfiltered in adversarial_examples:
        labels_source_unfiltered = model_source.predict(examples_unfiltered[:100])

        # Filter unsuccessful.
        examples_list = []
        no = -1
        for label_source_unfiltered, ground_truth in zip(labels_source_unfiltered, y_test[:100]):
            no += 1
            if np.argmax(label_source_unfiltered) != np.argmax(ground_truth):
                examples_list.append(examples_unfiltered[no])

        # Continue with filtered 100 example
        try:
            examples = np.array(examples_list[:100])
        except:
            examples = np.array(examples_list)

        mean_conferrability_scores = 0
        for example in examples:
            example = np.reshape(example, [1, 32, 32, 3])

            label_source = model_source.predict(example)

            # Calculate conferrability scores for each class.
            reference_prediction = tf.zeros_like(label_source)

            for model_reference in models_reference:
                prediction_r = model_reference(example)
                reference_prediction += prediction_r
            reference_prediction /= len(models_reference)

            surrogate_prediction = tf.zeros_like(label_source)

            for surrogate_model in models_surrogate:
                prediction_s = surrogate_model(example)
                surrogate_prediction += prediction_s
            surrogate_prediction /= len(models_surrogate)

            conferrability_score = np.max(surrogate_prediction * (tf.ones_like(reference_prediction) - reference_prediction))

            mean_conferrability_scores += conferrability_score

        mean_conferrability_scores /= len(examples)

        scores_for_e.append(mean_conferrability_scores)

    total_scores.append(scores_for_e)

bim_scores = [score[0] for score in total_scores]
pgd_scores = [score[1] for score in total_scores]
fgm_scores = [score[2] for score in total_scores]
cwlinf_scores = [score[3] for score in total_scores]
cem_scores = [score[4] for score in total_scores]

plt.plot(epsilon, bim_scores, '-x', label="BIM")
plt.plot(epsilon, pgd_scores, '-x', label="PGD")
plt.plot(epsilon, fgm_scores, '-x', label="FGM")
plt.plot(epsilon, cwlinf_scores, '-x', label="CWLInf")
plt.plot(epsilon, cem_scores, '-x', label="CEM")
plt.xlabel("Epsilon (L_inf)")
plt.ylabel("Conferrability")
plt.title("Conferrability Scores")
plt.legend()
plt.grid()
plt.savefig("fig_last.png")
