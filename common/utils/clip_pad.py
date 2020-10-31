import torch
from collections import Counter

def clip_pad_images(tensor, pad_shape, pad=0):
    """
    Clip clip_pad_images of the pad area.
    :param tensor: [c, H, W]
    :param pad_shape: [h, w]
    :return: [c, h, w]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    H, W = tensor.shape[1:]
    h = pad_shape[1]
    w = pad_shape[2]

    tensor_ret = torch.zeros((tensor.shape[0], h, w), dtype=tensor.dtype) + pad
    tensor_ret[:, :min(h, H), :min(w, W)] = tensor[:, :min(h, H), :min(w, W)]

    return tensor_ret


def clip_pad_boxes(tensor, pad_length, pad=0):
    """
        Clip boxes of the pad area.
        :param tensor: [k, d]
        :param pad_shape: K
        :return: [K, d]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    k = tensor.shape[0]
    d = tensor.shape[1]
    K = pad_length
    tensor_ret = torch.zeros((K, d), dtype=tensor.dtype) + pad
    tensor_ret[:min(k, K), :] = tensor[:min(k, K), :]

    return tensor_ret


def clip_pad_1d(tensor, pad_length, pad=0):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor_ret = torch.zeros((pad_length, ), dtype=tensor.dtype) + pad
    tensor_ret[:min(tensor.shape[0], pad_length)] = tensor[:min(tensor.shape[0], pad_length)]

    return tensor_ret


def clip_pad_2d(tensor, pad_shape, pad=0):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor_ret = torch.zeros(*pad_shape, dtype=tensor.dtype) + pad
    tensor_ret[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])] \
        = tensor[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])]

    return tensor_ret


def tokenize_and_pad_answer(batched_answer, m2_transformer_info_list):
    """
    Method for VQA answer generation task. Designed for being output with transformer decoder.
    """
    text_field = m2_transformer_info_list[0]
    batched_pre_answer = []
    for answer in batched_answer:
        preprocessed_answer = text_field.preprocess(answer) # In M2Transformer, before calling preprocess, the variable is expected to be 1 string (of a sentence - the caption)
        batched_pre_answer.append(preprocessed_answer)
    # preprocessed_answer = (preprocessed_answer,)
    assert(isinstance(batched_pre_answer[0], list))
    batched_processed_answer = text_field.process(batched_pre_answer) # In M2Transformer, before calling process, the variable is expected to be batched (a tuple of list, each list is a sentence)
    return batched_processed_answer