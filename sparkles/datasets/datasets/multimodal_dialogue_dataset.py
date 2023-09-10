import os
import random
import re
import traceback
import numpy as np
import torchvision
from torch.utils.data import Dataset
import json
from PIL import Image
import copy
import torch
from transformers import LlamaTokenizer
from sparkles.models.sparkleschat import init_llama_tokenizer

IMAGEHERE_OLD = "<Img><ImageHere></Img>"
IMAGE_LEN = 32  # image embedding size
IMAGEHERE_TAG = "<Img>" + "<ImageHere>" + "<ImagePad>" * (IMAGE_LEN - 1) + "</Img>"


def prepare_label_data(input_ids, instruction, text_tokenizer, max_length, is_train):
    label = copy.deepcopy(input_ids)
    if is_train:
        instruction_token = text_tokenizer(instruction,
            return_tensors=None, padding="do_not_pad", truncation=True, max_length=max_length)
        if instruction[-1] == " ":
            # if the last token is a space, the last token is 29871,
            # which will not appear when instruction is concatenated with target text in label ids
            # for example, instruction is "###Assistant: ", instruction_token["input_ids"][-1] is 29871,
            # full text is "###Assistant: hi", full text's ids will not contain 29871
            # so we need to remove the last token when it is a space
            instruction_token["input_ids"] = instruction_token["input_ids"][:-1]
        label = [-100] * len(instruction_token["input_ids"]) + label[len(instruction_token["input_ids"]):]
    return label


def prepare_text_data(text, text_tokenizer, max_length, is_train=True, end_sym="###"):
    """
    The method prepares text tensor
    """
    if is_train:
        text = f"{text}{end_sym}"
        text_tokenizer.padding_side = "right"
    else:
        text_tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    text_tensor = text_tokenizer(text, max_length=max_length, truncation=True, padding="do_not_pad", return_tensors=None)
    return text_tensor


class MultimodalDialogueDataset(Dataset):
    def __init__(self, vis_processor, text_processor, anno_path, img_root, root_dir, llama_model):
        self.text_processor = text_processor
        self.prompt_template = "###Human: {question}###Assistant: "
        self.llama_tokenizer, self.media_token_id, self.media_pad_id = init_llama_tokenizer(llama_model)

        self.end_sym = "###"
        self.max_length = 1024
        # self.max_length = 512
        self.is_train = True
        self.vis_processor = vis_processor
        self.anno_path = os.path.join(root_dir, anno_path)
        self.img_root = os.path.join(root_dir, img_root)
        with open(self.anno_path, 'r') as fr:
            self.cases = json.load(fr)

    def __len__(self):
        return len(self.cases)

    def getitem(self, index):
        try:
            case = self.cases[index]
            question = case["uniform_question"] if "uniform_question" in case else case["question"]
            instruction = case["instruction"] if "instruction" in case else self.prompt_template.format(question=question)
            instruction = instruction.replace(IMAGEHERE_OLD, IMAGEHERE_TAG)  # these tag embeddings will be replaced by image embeddings
            answer = case["fix_answer"] if "fix_answer" in case else case["answer"]
            tgt_imgs = case["tgt_imgs"]
            images = []
            for idx, img in enumerate(tgt_imgs):
                image_path = os.path.join(self.img_root, str(img["image_id"]) + ".jpg")
                image = Image.open(image_path).convert('RGB')
                images.append(image)

            image_list = [self.vis_processor(s).unsqueeze(0) for s in images]
            image_tensor = torch.cat(image_list, dim=0)
            text = instruction + answer if self.is_train else instruction
            text_tensor = prepare_text_data(text, self.llama_tokenizer, self.max_length, self.is_train, self.end_sym)
            label_tensor = prepare_label_data(text_tensor["input_ids"], instruction, self.llama_tokenizer,
                                              self.max_length, self.is_train)
            example = {
                "image": image_tensor,
                "input_ids": text_tensor["input_ids"],
                "attention_mask": text_tensor["attention_mask"],
                "labels": label_tensor,
            }
        except Exception as e:
            print("Error when getitem:", e)
            traceback.print_exc()
            example = None

        return example

    def __getitem__(self, index):
        res = self.getitem(index)
        while res is None:
            index = random.randrange(0, self.__len__())
            res = self.getitem(index)
        return res

    def collater(self, samples):
        image_list, input_id_list, attention_mask_list, labels_list = [], [], [], []
        for sample in samples:
            image_list.append(sample["image"])
            input_id_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])

        # We have to pad
        # the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels_list)
        self.llama_tokenizer.padding_side = "right"
        padding_side = self.llama_tokenizer.padding_side
        padded_labels = []
        for l in labels_list:
            remainder = [-100] * (max_label_length - len(l))
            if isinstance(l, list):
                l = l + remainder if padding_side == "right" else remainder + l
            elif padding_side == "right":
                l = np.concatenate([l, remainder]).astype(np.int64)
            else:
                l = np.concatenate([remainder, l]).astype(np.int64)
            padded_labels.append(l)

        padded_samples = self.llama_tokenizer.pad({"input_ids": input_id_list, "attention_mask": attention_mask_list,
                                                   "labels": padded_labels}, return_tensors="pt", padding="longest",)
        return {
            "image": torch.stack(image_list, dim=0),
            "input_ids": padded_samples["input_ids"],
            "attention_mask": padded_samples["attention_mask"],
            "labels": padded_samples["labels"],
        }