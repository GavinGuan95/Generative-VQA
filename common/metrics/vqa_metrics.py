import torch
from .eval_metric import EvalMetric


class LossLogger(EvalMetric):
    def __init__(self, output_name, display_name=None,
                 allreduce=False, num_replicas=1):
        self.output_name = output_name
        if display_name is None:
            display_name = output_name
        super(LossLogger, self).__init__(display_name, allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if self.output_name in outputs:
                self.sum_metric += float(outputs[self.output_name].mean().item())
            self.num_inst += 1


class SoftAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(SoftAccuracy, self).__init__('SoftAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            cls_logits = outputs['label_logits']
            label = outputs['label']
            bs, num_classes = cls_logits.shape
            batch_inds = torch.arange(bs, device=cls_logits.device)
            # print("in vqa_metrics.py")
            # print("--- 0 ---")
            # print(label)
            # print(label.shape)
            # if label.shape[1] < 10:
            #     dummy = 1
            # print("--- 1 ---")
            # print(batch_inds)
            # print("--- 2 ---")
            # print(cls_logits)
            # print("--- 3 ---")
            # print(cls_logits.argmax(1))
            # print("--- 4 ---")
            # print(label[batch_inds, cls_logits.argmax(1)])
            # print("--- 5 ---")
            self.sum_metric += float(label[batch_inds, cls_logits.argmax(1)].sum().item())
            self.num_inst += cls_logits.shape[0]


class DecoderAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(DecoderAccuracy, self).__init__('SoftAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            # cls_logits = outputs['label_logits']
            # label = outputs['label']
            # bs, num_classes = cls_logits.shape
            # batch_inds = torch.arange(bs, device=cls_logits.device)
            # self.sum_metric += float(label[batch_inds, cls_logits.argmax(1)].sum().item())
            # self.num_inst += cls_logits.shape[0]

            decoder_output = torch.argmax(outputs['decoder_output'], 2)
            # give all decoder output after <eos> the value of <pad>
            for dec_out in decoder_output:
                try:
                    eos_idx = (dec_out == 3).nonzero()[0, 0] # 3 indicates <eos>
                    dec_out[eos_idx+1:] = 1
                except:
                    dec_out[:] = 1
                # mark prediction as false if <unk> is predicted
                if 0 in dec_out:
                    dec_out[:] = 0


            tokenized_answer = outputs['tokenized_answer']
            match_or_not = torch.all(torch.eq(decoder_output, tokenized_answer), dim=1)
            self.sum_metric += match_or_not.sum()
            # self.sum_metric += 1
            self.num_inst += match_or_not.shape[0]

