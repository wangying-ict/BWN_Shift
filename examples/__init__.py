from collections import OrderedDict

import torch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def add_weight_decay(model, weight_decay, skip_keys):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                no_decay.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def gen_key_map(new_dict, old_dict):
    new_keys = [(k, v.size()) for (k, v) in new_dict.items()]
    ori_keys = [(k, v.size()) for (k, v) in old_dict.items()]
    key_map = OrderedDict()
    assert len(new_keys) == len(ori_keys)
    for i in range(len(new_keys)):
        if 'expand_' in new_keys[i][0] and ori_keys[i][1] != new_keys[i][1]:
            print('{}({}) is expanded to {}({})'.format(ori_keys[i][0], ori_keys[i][1],
                                                        new_keys[i][0], new_keys[i][1]))
        else:
            assert ori_keys[i][1] == new_keys[i][1], '{} != {}'.format(ori_keys[i][1], new_keys[i][1])
        key_map[new_keys[i][0]] = ori_keys[i][0]
    return key_map


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
