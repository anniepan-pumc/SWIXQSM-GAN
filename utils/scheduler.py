from torch.optim import lr_scheduler

def linear_scheduler(optimizer, epochs_warmup, epochs_anneal, verbose=True):
    def lambda_rule(epoch, epochs_warmup, epochs_anneal):
        if epoch < epochs_warmup:
            return 1.0

        return 1.0 - (epoch - epochs_warmup) / (epochs_anneal + 1)

    lr_fn = lambda epoch: lambda_rule(epoch, epochs_warmup, epochs_anneal)

    return lr_scheduler.LambdaLR(optimizer, lr_fn, verbose=verbose)

def get_scheduler(optimizer, scheduler, name):
    if scheduler is None:
        return None
    
    verbose = True

    if name == 'linear':
        return linear_scheduler(optimizer, 5, 5, verbose=verbose)

    if name == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if name == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer,'min')

    if name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, 100,1, eta_min= 240 * 5e-7 / 48)

    if name == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100,1, eta_min= 240 * 5e-7 / 48)

    raise ValueError("Unknown scheduler '%s'" % name)