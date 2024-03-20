import math
import pdb
from tacotron2.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from tacotron2.plotting_utils import plot_gate_outputs_to_numpy
import random

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    if epoch == 0:
        return
    lr = max(min_lr, init_lr * (decay_rate ** epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cosine_step_schedule(optimizer, epoch, max_epoch, init_lr, min_lr, decay_rate, step_decay):
    """combine cosine and step the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    # Step decay phase
    if epoch % step_decay == 0 and epoch > 0:
        lr = max(min_lr, lr * (decay_rate ** epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cosine_step_schedule_v2(optimizer, epoch, max_epoch, init_lr, min_lr, decay_rate, step_decay):
    """combine cosine and step the learning rate"""
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % step_decay == 0 and epoch > 0:
    # Step decay phase
        lr = max(min_lr, current_lr * (0.5 ** epoch))
    else:
        lr = (current_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def load_npz(path):
    f = np.load(path)
    m, s = f['mu'][:], f['sigma'][:]
    f.close()
    return m, s

def save_models(netG, netD, netC, optG, optD, epoch, save_path):
    state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
            'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
            'epoch': epoch}
    torch.save(state, '%s/state_epoch_%03d.pth' % (save_path, epoch))

def truncated_noise(batch_size=1, dim_z=100, truncation=1., seed=None):
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values
def get_fix_data(train_dl, test_dl, model, text_encoder, args):
    fixed_image_train, fixed_sent_train = get_one_batch_data(train_dl, text_encoder, args)
    fixed_image_test, fixed_sent_test = get_one_batch_data(test_dl, text_encoder, args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    noise = truncated_noise(fixed_image.size(0), args["z_dim"], args["trunc_rate"])
    fixed_noise = torch.tensor(noise, dtype=torch.float).to("cuda")
    return fixed_image, fixed_sent, fixed_noise

def get_one_batch_data(dataloader, text_encoder, args):
    data = next(iter(dataloader))
    images, image_netg, labels, decoder_target, attention_mask, phone_len, item_ids = data
    labels = labels.to("cuda")
    phone_len = phone_len.to("cuda")
    with torch.no_grad():
        hidden = text_encoder.init_hidden(labels.size(0))
        _, sent_emb = text_encoder(labels, phone_len, hidden, one_hot=False)
    return image_netg, sent_emb

def log_auditory_feedback(y_target, y_pred, epoch, logger):
    mel_outputs, gate_outputs, alignments = y_pred
    mel_targets, gate_targets, alignments_gt = y_target

    # plot alignment, mel target and predicted, gate target and predicted
    idx = random.randint(0, alignments.size(0) - 1)
    logger.add_image(
        "alignment",
        plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
        epoch, dataformats='HWC')
    logger.add_image(
        "alignment_target",
        plot_alignment_to_numpy(alignments_gt[idx].data.cpu().numpy().T),
        epoch, dataformats='HWC')
    logger.add_image(
        "mel_target",
        plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
        epoch, dataformats='HWC')
    logger.add_image(
        "mel_predicted",
        plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
        epoch, dataformats='HWC')
    logger.add_image(
        "gate",
        plot_gate_outputs_to_numpy(
            gate_targets[idx].data.cpu().numpy(),
            torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
        epoch, dataformats='HWC')

def log_auditory_feedback_v2(y_target, y_pred, epoch, logger):
    mel_outputs, gate_outputs, alignments = y_pred
    mel_targets, gate_targets, alignments_gt = y_target

    # plot alignment, mel target and predicted, gate target and predicted
    idx = random.randint(0, alignments.size(0) - 1)
    logger.add_image(
        "alignment",
        plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
        epoch, dataformats='HWC')
    logger.add_image(
        "alignment_target",
        plot_alignment_to_numpy(alignments_gt[idx].data.cpu().numpy().T),
        epoch, dataformats='HWC')
    logger.add_image(
        "mel_target",
        plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
        epoch, dataformats='HWC')
    logger.add_image(
        "mel_predicted",
        plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
        epoch, dataformats='HWC')
    logger.add_image(
        "gate",
        plot_gate_outputs_to_numpy(
            gate_targets[idx].data.cpu().numpy(),
            torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
        epoch, dataformats='HWC')

def log_auditory_feedback_v3(y_target, y_pred, epoch, logger):
    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = y_pred
    mel_targets, gate_targets = y_target

    # plot alignment, mel target and predicted, gate target and predicted
    idx = random.randint(0, alignments.size(0) - 1)
    logger.add_image(
        "alignment",
        plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
        epoch, dataformats='HWC')
    logger.add_image(
        "mel_target",
        plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
        epoch, dataformats='HWC')
    logger.add_image(
        "mel_predicted",
        plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
        epoch, dataformats='HWC')
    logger.add_image(
        "mel_predicted_postnet",
        plot_spectrogram_to_numpy(mel_outputs_postnet[idx].data.cpu().numpy()),
        epoch, dataformats='HWC')
    logger.add_image(
        "gate",
        plot_gate_outputs_to_numpy(
            gate_targets[idx].data.cpu().numpy(),
            torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
        epoch, dataformats='HWC')
