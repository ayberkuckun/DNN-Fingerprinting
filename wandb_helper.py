import wandb


def start(id, dataset, model, surrogate_no, reference_no, use_saved, epsilon, p, threshold):
    wandb.login()

    if id is None:
        wandb.init(project="Federated-Finger-Printing", entity="ayberk")
    else:
        wandb.init(project="Federated-Finger-Printing", entity="ayberk", resume=True, id=id)

    wandb.config = {
        "dataset": dataset,
        "model": model,
        "surrogate_no": surrogate_no,
        "reference_no": reference_no,
        "use_saved": use_saved,
        "initial_learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 32,
        "epsilon": epsilon,
        "Decision threshold": p,
        "Conferrability threshold": threshold

    }

    # wandb.define_metric("gen_G_loss", summary="min")
    # wandb.log({"loss": loss})
    # Optional
    # wandb.watch(model)
    # maybe report working adversarial examples/fingerprints with some threshold
