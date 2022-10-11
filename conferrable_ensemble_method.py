import argparse
import csv
import random

import h5py
import numpy as np
from keras_preprocessing.image import save_img
from matplotlib import pyplot as plt

from cem_helper_func import *
from constants import *

import time

import wandb_helper


# todo rewrite model creation
# todo mixed precision, data augmentation, jix function
def conferrable_ensemble_method(dataset="cifar10", model="ResNet20v1", surrogate_no=2, reference_no=2, use_saved=False):
    # # Dataset
    # # dataset = "cifar10"
    # # dataset = "imagenet32"
    # dataset = "mnist"
    # dataset = "imagenet32-pre"
    #
    # # Model
    # # model = "ResNet50v2"
    # # model = "MobileNetV2"
    # model = "ResNet20v1"
    # model = "ResNet50v2-pre"
    #
    # # Surrogate and Reference models
    # surrogate_no = 16
    # reference_no = 16  # Original 18

    # Current date.
    time_str = time.strftime("%Y_%m_%d_%H")

    # Start wandb for experiment tracking.
    wandb_helper.start(id=None, dataset=dataset, model=model, surrogate_no=surrogate_no,
                       reference_no=reference_no, use_saved=use_saved, epsilon=epsilon, p=p,
                       threshold=threshold)

    (x_train, y_train), (x_test, y_test) = prepare_data(dataset=dataset, subtract_pixel_mean=True, shuffle=True, seed=None)

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Model Preparation
    source_model, reference_models, surrogate_models = prepare_models(x_train, y_train, x_test, y_test,
                                                                      dataset,
                                                                      model,
                                                                      input_shape=input_shape,
                                                                      use_saved=use_saved,
                                                                      surrogate_no=surrogate_no,
                                                                      reference_no=reference_no)

    if not use_saved:
        return None

    source_model.trainable = False

    for reference_model in reference_models:
        reference_model.trainable = False

    for surrogate_model in surrogate_models:
        surrogate_model.trainable = False

    for e in epsilon:
        print(f"Epsilon: {e}")

        fingerprint = []
        fingerprint_labels = []
        fingerprint_ground_truth = []
        fingerprint_stats = []
        for example_no, example in enumerate(x_train[:10]):
            if example_no == 10:
                break

            ground_truth = np.argmax(y_train[example_no])
            print(f"Example No: {example_no} Label: {ground_truth}")

            example = tf.reshape(example, [1, 32, 32, 3])
            initial_source_prediction = source_model(example)

            if np.argmax(initial_source_prediction) != ground_truth:
                print("Example initially predicted wrong. Skipping..")
                continue

            adversarial_example = tf.Variable(example)

            i = 0
            not_learning = 0
            while i < 100:
                with tf.GradientTape() as tape:
                    tape.watch(adversarial_example)

                    source_prediction = source_model(adversarial_example)

                    reference_prediction = tf.zeros_like(initial_source_prediction)

                    # Dropout = 0.3
                    model_selection = np.random.binomial(1, 0.7, reference_no)
                    mask_r = np.ma.make_mask(model_selection)
                    reference_models_dropped = np.array(reference_models)[mask_r]

                    for reference_model in reference_models_dropped:
                        prediction_r = reference_model(adversarial_example)
                        reference_prediction += prediction_r
                    reference_prediction /= len(reference_models_dropped)

                    surrogate_prediction = tf.zeros_like(initial_source_prediction)

                    # Dropout = 0.3
                    model_selection = np.random.binomial(1, 0.7, surrogate_no)
                    mask_s = np.ma.make_mask(model_selection)
                    surrogate_models_dropped = np.array(surrogate_models)[mask_s]

                    for surrogate_model in surrogate_models_dropped:
                        prediction_s = surrogate_model(adversarial_example)
                        surrogate_prediction += prediction_s
                    surrogate_prediction /= len(surrogate_models_dropped)

                    ensemble_output = surrogate_prediction * (tf.ones_like(reference_prediction) - reference_prediction)

                    first_loss_input = tf.reshape(tf.slice(ensemble_output, [0, tf.argmax(
                        tf.reshape(source_prediction, [100]), output_type=tf.dtypes.int32)], [1, 1]), ())

                    first_loss = alpha * tf.reshape(tf.keras.losses.categorical_crossentropy(
                        [1.0, 0.0], [first_loss_input, 1.0 - first_loss_input]), ())

                    second_loss = beta * tf.reshape(
                        tf.keras.losses.categorical_crossentropy(initial_source_prediction,
                                                                 tf.clip_by_value(source_prediction, 1e-7, 1. - 1e-7)),
                        ())
                    # tf.keras.losses.categorical_crossentropy(initial_source_prediction, source_prediction), ())

                    third_loss = gamma * tf.reshape(
                        tf.keras.losses.categorical_crossentropy(source_prediction, surrogate_prediction), ())

                    loss = first_loss - second_loss + third_loss

                # try:
                #     print(f"\nIteration : {i} Loss: {loss:.4f} lr: {round(optimizer1.lr(i).numpy(), 5)}")
                # except:
                #     print(f"\nIteration : {i} Loss: {loss:.4f}")
                #
                # print(f"\t First Loss: {first_loss:.4f}")
                # print(f"\t Second Loss: {second_loss:.4f}")
                # print(f"\t Third Loss: {third_loss:.4f}\n")

                grads = tape.gradient(loss, adversarial_example)

                # Update ratio check
                ratio = np.linalg.norm((optimizer1.lr.numpy() * grads).ravel())/np.linalg.norm(adversarial_example.ravel())
                print("Ratio of Updates/Weights: ", ratio)

                # if ratio > 1e-2 or ratio < 1e-4:
                #     print("Ratio of Updates/Weights is wrong: ", ratio)

                optimizer1.apply_gradients(zip([grads], [adversarial_example]))

                # Apply clipping around e ball
                perturbations_raw = adversarial_example - example

                # For infinity norm, normally it should be (max of absolute sums of rows < e)
                perturbations_projected = tf.map_fn(lambda x: np.clip(x, -e, e), perturbations_raw)

                adversarial_example_var = example + perturbations_projected

                adversarial_example = tf.Variable(adversarial_example_var)

                i += 1

            # Conferrability is calculated using all models.
            reference_prediction_last = tf.zeros_like(initial_source_prediction)

            for reference_model in reference_models:
                prediction_r = reference_model(adversarial_example)
                reference_prediction_last += prediction_r
            reference_prediction_last /= len(reference_models)

            surrogate_prediction_last = tf.zeros_like(initial_source_prediction)

            for surrogate_model in surrogate_models:
                prediction_s = surrogate_model(adversarial_example)
                surrogate_prediction_last += prediction_s
            surrogate_prediction_last /= len(surrogate_models)

            ensemble_output_last = surrogate_prediction_last * (tf.ones_like(reference_prediction_last) - reference_prediction_last)

            # Add example if it's adversarial to source model.
            pred = source_model(adversarial_example)
            label = np.argmax(pred)
            if label != ground_truth:

                # Add example to fingerprint if conferrability score is high enough
                conferrability_score = tf.reshape(tf.reduce_max(ensemble_output_last), ()).numpy()
                print(f"Conferrability Score: {conferrability_score:.2f}")

                if conferrability_score >= threshold:
                    fingerprint.append(adversarial_example.value().numpy())
                    fingerprint_labels.append(label)
                    fingerprint_ground_truth.append(ground_truth)
                    fingerprint_stats.append(
                        {
                            "source_label": f"{label} - {100 * np.max(pred):.2f}%",
                            "average_surrogate_label": f"{np.argmax(surrogate_prediction):.2f} - {100 * np.max(surrogate_prediction):.2f}%",
                            "average_reference_label": f"{np.argmax(reference_prediction):.2f} - {100 * np.max(reference_prediction):.2f}%",
                            "real_label": ground_truth,
                            "conferrability_score": round(conferrability_score, 2)
                        }
                    )

                    # plt.figure()
                    # plt.imshow(tf.reshape(example, [32, 32, 3]))
                    # plt.show()
                    # plt.imshow(tf.reshape(adversarial_example.value(), [32, 32, 3]))
                    # plt.show()

        # Save fingerprint stats.
        keys = fingerprint_stats[0].keys()

        stat_path = f'stats/{dataset}/' + time_str
        if not os.path.exists(stat_path):
            os.makedirs(stat_path)

        with open(f'{stat_path}/fingerprint_stats.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(fingerprint_stats)

        # Save generated fingerprints.
        fingerprint = np.reshape(np.array(fingerprint), (-1, 32, 32, 3))
        fingerprint_labels = np.reshape(np.array(fingerprint_labels), (len(fingerprint_labels), 1))
        fingerprint_ground_truth = np.reshape(np.array(fingerprint_ground_truth), (len(fingerprint_ground_truth), 1))

        dataset_path = f'fingerprints/{dataset}/e_{e}/' + time_str
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        with h5py.File(f'{dataset_path}/fingerprint.h5', 'w') as h5f:
            h5f.create_dataset('fingerprint', data=fingerprint)
            h5f.create_dataset('fingerprint_labels', data=fingerprint_labels)
            h5f.create_dataset('fingerprint_ground_truth', data=fingerprint_ground_truth)


if __name__ == "__main__":
    # Arguments parsing
    arguments_parser = argparse.ArgumentParser(description="Run the Exp.")
    arguments_parser.add_argument("dataset", type=str)
    arguments_parser.add_argument("model", type=str)
    arguments_parser.add_argument("surrogate_no", type=int)
    arguments_parser.add_argument("reference_no", type=int)
    arguments_parser.add_argument("--use_saved", help="Use saved models", action="store_true")
    args = arguments_parser.parse_args()

    conferrable_ensemble_method(args.dataset, args.model, args.surrogate_no, args.reference_no, args.use_saved)
