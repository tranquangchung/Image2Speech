import pdb

from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets, loss_type='L1'):
        mel_target, gate_target = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        if loss_type == 'L1':
            mel_loss = nn.L1Loss()(mel_out, mel_target)
        elif loss_type == 'MSE':
            mel_loss = nn.MSELoss()(mel_out, mel_target)
        else:
            raise ValueError('loss_type should be either L1 or MSE')
        return mel_loss + gate_loss

class Tacotron2Loss_V2(nn.Module):
    def __init__(self):
        super(Tacotron2Loss_V2, self).__init__()

    def forward(self, model_output, targets, loss_type='L1'):
        mel_target, gate_target, alignments_target = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)
        if loss_type == 'L1':
            mel_loss = nn.L1Loss()(mel_out, mel_target)
        elif loss_type == 'MSE':
            mel_loss = nn.MSELoss()(mel_out, mel_target)
        else:
            raise ValueError('loss_type should be either L1 or MSE')
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        # alignments_loss = nn.MSELoss()(alignments, alignments_target)
        return mel_loss + gate_loss

class Tacotron2LossV3(nn.Module):
    def __init__(self):
        super(Tacotron2LossV3, self).__init__()

    def forward(self, model_output, targets, loss_type='L1'):
        mel_target, gate_target = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_outputs_postnet, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        if loss_type == 'L1':
            mel_loss = nn.L1Loss()(mel_out, mel_target) + nn.L1Loss()(mel_outputs_postnet, mel_target)
        elif loss_type == 'MSE':
            mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_outputs_postnet, mel_target)
        else:
            raise ValueError('loss_type should be either L1 or MSE')
        return mel_loss + gate_loss