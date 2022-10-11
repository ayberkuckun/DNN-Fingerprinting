import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

from paper.CEM.constants import p

# Dataset
dataset = "cifar10"
# dataset = "imagenet32"
# dataset = "imagenet32-pre"
# dataset = "mnist"

# Model
# model = "ResNet50v2"
# model = "MobileNetV2"
model = "ResNet20v1"
# model = "ResNet50v2-pre"

e = 0.025
date = "01_29"

path_source = os.path.join(os.getcwd(), f'saved_models/{dataset}/source/0/{model}_model.h5')
source_model = keras.models.load_model(path_source)

path_surrogate = os.path.join(os.getcwd(), f'saved_models/{dataset}/surrogate/15/{model}_model.h5')
surrogate_model = keras.models.load_model(path_surrogate)

path_reference = os.path.join(os.getcwd(), f'saved_models/{dataset}/reference/15/{model}_model.h5')
reference_model = keras.models.load_model(path_reference)

with h5py.File(f'fingerprints/e_{e}/fingerprint_train.h5', 'r') as h5file:
    fingerprint = np.array(h5file["fingerprint"])
    fingerprint_labels = np.array(h5file["fingerprint_labels"])
    fingerprint_ground_truth = np.array(h5file["fingerprint_ground_truth"])

fingerprint = np.reshape(np.array(fingerprint), (-1, 32, 32, 3))

# delete_array = []
# for no, (l, g) in enumerate(zip(fingerprint_labels, fingerprint_ground_truth)):
#     if l == g:
#         delete_array.append(no)
#
# fingerprint = np.delete(fingerprint, delete_array, axis=0)
# fingerprint_labels = np.delete(fingerprint_labels, delete_array, axis=0)
# fingerprint_ground_truth = np.delete(fingerprint_ground_truth, delete_array, axis=0)

# for image in fingerprint:
#     plt.figure()
#     plt.imshow(image)
#     plt.show()

# Confusion Matrix.
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
confusion_matrix = confusion_matrix(
    fingerprint_ground_truth,
    fingerprint_labels,
    # labels=labels,
    normalize='all'
)

fig, ax = plt.subplots(figsize=(10, 4))

ConfusionMatrixDisplay(confusion_matrix*100, labels).plot(include_values=True, cmap='magma', ax=ax)  # ax=ax
# Get the images on an axis
im = ax.images

# Assume colorbar was plotted last one plotted last
cbar = im[-1].colorbar
cbar.set_label('Normalized % Ratio')
plt.tight_layout()
plt.show()

# Test for target model verification
source_labels = source_model.predict(fingerprint)
target_labels = surrogate_model.predict(fingerprint)
reference_labels = reference_model.predict(fingerprint)

# Conferrable Adversarial Example Accuracy (CAEAcc)
verified_examples = 0
for source_label, target_label in zip(source_labels, target_labels):
    if np.argmax(source_label) == np.argmax(target_label):
        verified_examples += 1
CAEAcc_surr = verified_examples / len(source_labels)
print(f"Surrogate CAEAcc: {CAEAcc_surr}")

# Apply decision threshold
if CAEAcc_surr < p:
    print("Model is a Reference Model.")
else:
    print("Model is a Surrogate Model")

verified_examples = 0
for source_label, reference_label in zip(source_labels, reference_labels):
    if np.argmax(source_label) == np.argmax(reference_label):
        verified_examples += 1
CAEAcc_ref = verified_examples / len(source_labels)
print(f"Reference CAEAcc: {CAEAcc_ref}")

# Apply decision threshold
if CAEAcc_ref < p:
    print("Model is a Reference Model.")
else:
    print("Model is a Surrogate Model")
