import numpy as np
import torch

from medzoo.lib.utils.general import prepare_input
from medzoo.lib.visual3D_temp.BaseWriter import TensorboardWriter


class Trainer:
    """
    Trainer class
    """

    def __init__(
        self,
        args,
        model,
        criterion,
        optimizer,
        train_data_loader,
        val_criterion=None,
        valid_data_loader=None,
        lr_scheduler=None,
    ):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_criterion = val_criterion or criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        # self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = 10
        # self.terminal_show_freq = self.args.terminal_show_freq
        self.terminal_show_freq = 100000
        self.start_epoch = 1

    def training(self):
        for epoch in range(self.start_epoch, self.args.nEpochs):
            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = (
                self.writer.data["val"]["loss"] / self.writer.data["val"]["count"]
            )

            if self.args.save is not None and ((epoch + 1) % self.save_frequency):
                self.model.save_checkpoint(
                    self.args.save, epoch, val_loss, optimizer=self.optimizer
                )

            self.writer.write_end_of_epoch(epoch)

            self.writer.reset("train")
            self.writer.reset("val")

    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx, input_tuple in enumerate(self.train_data_loader):

            self.optimizer.zero_grad()

            input_tensor, target = prepare_input(
                input_tuple=input_tuple, args=self.args
            )
            # print(input_tensor.size())
            # torch.Size([8, 3, 48, 48, 48])
            input_tensor.requires_grad = True
            output = self.model(input_tensor)

            # output = output.type(torch.LongTensor)
            """
            
            torch.Size([8, 3, 64, 64, 64]) torch.cuda.FloatTensor tensor(-4.6913, device='cuda:0', grad_fn=<MinBackward1>) tensor(6.2169, device='cuda:0', grad_fn=<MaxBackward1>)
            torch.Size([8, 64, 64, 64]) torch.cuda.LongTensor tensor(0, device='cuda:0') tensor(2, device='cuda:0')

            """
            target = target.type(torch.LongTensor).cuda()
            # print(output.size(), output.type(), torch.min(output).item(), torch.max(output).item())
            # print(target.size(), target.type(), torch.min(target).item(), torch.max(target).item())
            # print(self.criterion)

            # loss_dice, per_ch_score = self.criterion(output, target)
            loss_dice = self.criterion(output, target)
            per_ch_score = [loss_dice.item()]
            # print(loss_dice)
            # print(per_ch_score)

            loss_dice.backward()
            self.optimizer.step()

            self.writer.update_scores(
                batch_idx,
                loss_dice.item(),
                per_ch_score,
                "train",
                epoch * self.len_epoch + batch_idx,
            )

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, "train")

        self.writer.display_terminal(self.len_epoch, epoch, mode="train", summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(
                    input_tuple=input_tuple, args=self.args
                )
                input_tensor.requires_grad = False

                output = self.model(input_tensor)
                # loss, per_ch_score = self.criterion(output, target)
                loss = self.criterion(output, target)
                per_ch_score = [loss.item()]

                self.writer.update_scores(
                    batch_idx,
                    loss.item(),
                    per_ch_score,
                    "val",
                    epoch * self.len_epoch + batch_idx,
                )

        self.writer.display_terminal(
            len(self.valid_data_loader), epoch, mode="val", summary=True
        )
