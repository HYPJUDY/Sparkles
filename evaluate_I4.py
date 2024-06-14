import copy
import json
import argparse
import os
import random
import re
import numpy as np
import torch.backends.cudnn as cudnn
import more_itertools
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.call_gpt_api import openai_chat_create
from dataset.visualize_data_in_html import visualize_dialogues_in_html, visualize_VLtasks_in_html
from sparkles.common.config import Config
from sparkles.common.registry import registry
from sparkles.conversation.conversation_sparkleschat import Chat, CONV_VISION

import torch
from PIL import Image



class MiniGPT4:
    def __init__(self, cfg, gpu_ids) -> None:
        model_config = cfg.model_cfg
        devices = ['cuda:{}'.format(id) for id in gpu_ids]
        model_config.device_8bit = devices[0]
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(devices[0])
        if len(devices) > 1:  # If there is more than one device, use DataParallel
            model = nn.DataParallel(model, device_ids=devices)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device = devices[0])

    def generate(self, prompt, img_paths=None, num_beams=1, temperature=1.0, *kargs):
        chat_state = CONV_VISION.copy()
        img_list = []
        if img_paths is not None:
            for img_path in img_paths:
                image = Image.open(img_path).convert('RGB')
                self.chat.upload_img(image, chat_state, img_list)
        self.chat.ask(prompt, chat_state)
        llm_message = self.chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=1000,
                                    max_length=2000)[0]
        return llm_message


class IDataset(Dataset):
    def __init__(
            self, annoation, task_instructions, img_dir,
    ):
        self.img_dir = img_dir
        self.annotation = annoation
        self.task_instructions = task_instructions

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        task_instruction = self.task_instructions[ann['task_instruction_id']]
        context = task_instruction + ann['task_instance']['context']
        raw_img_list = []
        if 'choice_list' in ann['task_instance'].keys():
            choice_str = 'Choice list:[\'' + '\', \''.join(ann['task_instance']['choice_list']) + '\']. Your answer is:'
            context += choice_str
        for i in range(len(ann['task_instance']['images_path'])):
            rmv_txt = '{image#%d}' % (i + 1)
            rmv_tbl = '{table#%d}' % (i + 1)
            context = context.replace(rmv_txt, '<Img><ImageHere></Img>')
            context = context.replace(rmv_tbl, '<Img><ImageHere></Img>')
        for p in ann['task_instance']['images_path']:
            img_path = os.path.join(self.img_dir, p)
            raw_img_list.append(img_path)
        return {
            "sample_id": ann['sample_id'],
            "context": context,
            "raw_img_list": raw_img_list,
            "response": str(ann['response'])
        }


def collate_fn(batch):
    batch_data = {}
    batch_data['sample_id'] = [sample['sample_id'] for sample in batch]
    batch_data['context'] = [sample['context'] for sample in batch]
    batch_data['raw_img_list'] = [sample['raw_img_list'] for sample in batch]
    batch_data['response'] = [sample['response'] for sample in batch]

    return batch_data


def split_data(data):
    data_dict = {}
    for d in data:
        n_img = len(d['task_instance']['images_path'])
        if n_img in data_dict:
            data_dict[n_img].append(d)
        else:
            data_dict[n_img] = [d]
    return data_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark prediction")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument('--batch-image', type=int, required=False, default=30)
    parser.add_argument('--i4-dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--result-dir",type=str,required=True)
    parser.add_argument("--sparkles_root", type=str, default='/path/to/Sparkles/', help="data root")
    parser.add_argument("--seed", type=int, default=2023, help="")
    parser.add_argument("--num-beams", type=int, default=1, help="num beams for beam search.")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args



def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == "__main__":
    args = parse_args()
    setup_seeds(args.seed)

    num_beams = args.num_beams
    temperature = args.temperature
    # dataset_root = f"{args.sparkles_root}/evaluation/{args.dataset}"

    cfg = Config(args)
    ckpt_path = cfg.model_cfg.ckpt
    beam = f"_beam{num_beams}" if num_beams > 1 else ""
    # model_name = f"{ckpt_path.split('/')[-3]}_{ckpt_path.split('/')[-2]}_{ckpt_path.split('/')[-1][:-4]}{beam}"
    # results_dir = f"{dataset_root}/results"
    # results_path_name = f"{results_dir}/{args.dataset}_{model_name}"
    #
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)

    model = MiniGPT4(cfg, [args.gpu_id])
    # answer_path = f"{results_path_name}.json"
    # if not os.path.exists(os.path.dirname(answer_path)):
    #     os.makedirs(os.path.dirname(answer_path))

    i4_dir = args.i4_dir
    dataset_name = args.dataset
    batch_image = args.batch_image
    dataset_dir = os.path.join(i4_dir, dataset_name, 'core')
    img_dir = os.path.join(dataset_dir, 'images')
    output_dir = os.path.join(args.result_dir, dataset_name)
    # model_name = args.result_dir.split('/')[-1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    core_annotation = json.load(open(os.path.join(dataset_dir, 'core.json'), 'r'))
    prediction_results = []
    data_dict = split_data(core_annotation['data'])

    for n_img, sub_data in data_dict.items():
        E = IDataset(sub_data, core_annotation['metadata']['task_instruction'], img_dir)
        data_loader = torch.utils.data.DataLoader(dataset=E, batch_size=max(int(batch_image / n_img), 1),
                                                  shuffle=False, num_workers=0, collate_fn=collate_fn)
        for i, samples in enumerate(tqdm(data_loader)):
            pred_responses = []
            for j in range(len(samples['context'])):
                pred_response = model.generate(img_paths=samples['raw_img_list'][j], prompt=samples['context'][j],
                                      num_beams=num_beams, temperature=temperature)
                pred_responses.append(pred_response)
            # pred_responses = chat.batch_answer(batch_raw_img_list=samples['raw_img_list'],
            #                                    batch_context=samples['context'], max_length=5000)
            for sid, gt, p in zip(samples['sample_id'], samples['response'], pred_responses):
                if torch.is_tensor(sid):
                    sid = sid.item()
                prediction_result = {'sample_id': sid, 'pred_response': p, 'gt_response': gt}
                prediction_results.append(prediction_result)
                print(prediction_result)

    with open(os.path.join(output_dir,'pred.json'),'w',encoding='utf8') as f:
        json.dump(prediction_results,f,indent=4,ensure_ascii=False)