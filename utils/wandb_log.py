import wandb

class Wandb:
    def __init__(self, project, flag, name, config):
        self.flag = flag
        if self.flag:
            wandb.init(project=project, name=name, entity="boostcv16", config=config)
    
    def write_log(self, train_loss, train_acc, current_lr):
        if self.flag:
            wandb.log({"Train/loss": train_loss, "Train/accuracy": train_acc, "learning rate": current_lr})

    def write_log2(self, epoch, current_lr, val_loss, val_acc, f1):
        if self.flag:
            wandb.log({"epoch": epoch, "learning rate": current_lr, "Val/loss": val_loss,"Val/accuracy": val_acc, "F1 Score": f1})


    def write_log3(self, best_val_acc, best_f1):
        if self.flag:
            wandb.log({"best Val acc": best_val_acc, "best F1": best_f1})

    def watch(self, model):
        if self.flag:
            wandb.watch(model)