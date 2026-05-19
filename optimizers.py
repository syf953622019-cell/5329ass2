import torch.optim as optim
from sam import SAM


def get_optimizer(model, optimizer_name, lr, weight_decay=5e-4, rho=0.05):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "sam":
        optimizer = SAM(
            model.parameters(),
            base_optimizer=optim.SGD,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            rho=rho,
        )

    else:
        raise ValueError("optimizer_name must be one of: sgd, adamw, sam")

    return optimizer
