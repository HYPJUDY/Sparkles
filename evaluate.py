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


class BISONDataset(Dataset):
    def __init__(
            self,
            dataset_root,
            data_from=0,
            data_to=50,
    ):
        self.options = "IMAGE#1/IMAGE#2"
        self.prompt_template = """Carefully examine the two similar images of IMAGE#1<Img><ImageHere></Img> and IMAGE#2<Img><ImageHere></Img>.
Given the following caption, you must select which of two images best matches the caption.
The caption is: '{}'.
This task requires fine-grained visual reasoning between the caption and each image. Let's think step by step.
Please start your response with "Let's think step by step." and end with "Therefore, the answer (IMAGE#1 or IMAGE#2) is"."""

        self.image_dir = f"{dataset_root}/images"
        data_path = f"{dataset_root}/annotations/sparkles_evaluation_bison_annotations.json"
        with open(data_path, 'r') as fr:
            self.data = json.load(fr)
        self.data = self.data[data_from:data_to]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['caption']
        prompt = self.prompt_template.format(sentence)
        label = 'IMAGE#1' if item['true_image_id'] == item['image_candidates'][0]['image_id'] else 'IMAGE#2'
        img_path1 = os.path.join(self.image_dir, item['image_candidates'][0]['image_filename'])
        img_path2 = os.path.join(self.image_dir, item['image_candidates'][1]['image_filename'])
        return {
            "img_paths": [img_path1, img_path2],
            "sentence": sentence,
            "prompt": prompt,
            "label": label,
            "evaluation_id": item['evaluation_id'],
        }


class NLVR2Dataset(Dataset):
    def __init__(
            self,
            dataset_root,
            data_from=50,
            data_to=100,
    ):
        self.options = "TRUE/FALSE"
        self.prompt_template = """Carefully examine a pair of images: the left IMAGE#1<Img><ImageHere></Img> and the right IMAGE#2<Img><ImageHere></Img>.
Determine whether the following statement is true about the pair of images:
'{}'
Jointly reasoning about the statement grounded in IMAGE#1 and IMAGE#2.
The task requires compositional joint reasoning, including about quantities, comparisons, and relations. Let's think step by step.
Please start your response with "Let's think step by step." and end with "Therefore, the answer (TRUE or FALSE) is"."""

        self.image_dir = f"{dataset_root}/images"
        data_path = f"{dataset_root}/annotations/sparkles_evaluation_nlvr2_annotations.json"
        with open(data_path, 'r') as fr:
            self.data = json.load(fr)
        self.data = self.data[data_from:data_to]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['sentence']
        prompt = self.prompt_template.format(sentence)
        label = item['label']
        img_path1 = os.path.join(self.image_dir, f"{item['identifier'][:-2]}-img0.png")
        img_path2 = os.path.join(self.image_dir, f"{item['identifier'][:-2]}-img1.png")
        return {
            "img_paths": [img_path1, img_path2],
            "sentence": sentence,
            "prompt": prompt,
            "label": label,
            "evaluation_id": item['evaluation_id'],
        }


def extract_answer(response, options="TRUE/FALSE"):
    options = [option.lower() for option in options.split("/")]
    response = response.lower()
    # required_words = ["Let's think step by step", "Therefore"]
    required_words = ["Therefore"]
    for word in required_words:
        if word.lower() not in response:
            return None
    # Find the last occurrence of "Therefore"
    answer = response[response.rfind("therefore") + len("therefore"):]
    answer = answer.strip().lower()
    answer = answer.replace("answer (true or false) is", "answer is")
    answer = answer.replace("image #1", "image#1").replace("image #2", "image#2")
    answer = answer.replace("answer (image#1 or image#2) is", "answer is")
    answer = answer.replace(f"not {options[0]}", options[1]).replace(f"not {options[0]}", options[1])
    # "Therefore, we cannot determine if the statement is true or false" -> None
    # "Therefore, the statement "A man is sitting" cannot be true and is FALSE." -> FALSE
    # "Therefore, the answer is TRUE for Image#2 and FALSE for Image#1. In conclusion, there is one true and one false statement in this comparison." -> NONE
    # "Therefore, IMAGE#2 is not the correct match." -> None
    # "Therefore, the answer is IMAGE#1, because it more closely matches the caption than IMAGE#2." -> IMAGE#1
    if not (f"the answer is {options[0]}," in answer or
            f"the answer is {options[1]}," in answer or
            f"the answer is {options[0]}." in answer or
            f"the answer is {options[1]}." in answer or
            f"the answer is {options[0]}:" in answer or
            f"the answer is {options[1]}:" in answer or
            f"the answer is {options[0]} because" in answer or
            f"the answer is {options[1]} because" in answer) and \
            (options[0] in answer and options[1] in answer):
            # if f"neither {options[0]} nor {options[1]}" in answer or \
            # f"{options[0]} or {options[1]}" in answer or \
            # f"{options[0]} and {options[1]}" in answer:
        return None
    # extract the closest matched option in answer
    pos0 = answer.find(options[0])
    pos1 = answer.find(options[1])
    if pos0 == -1 and pos1 == -1:
        return None
    if pos0 == -1:
        return options[1]
    if pos1 == -1:
        return options[0]
    return options[0] if pos0 < pos1 else options[1]


def calculate_score(predictions):
    if len(predictions) == 0:
        return 0
    correct = 0
    for prediction in predictions:
        if prediction['predict'].lower() == prediction['label'].lower():
            correct += 1
    return correct / len(predictions)


def evaluate_pairs(model, dataset, answer_path, options="TRUE/FALSE", batch_size=1, num_beams=1, temperature=1.0):
    predictions = []
    predict_num = 0
    for batch in more_itertools.chunked(tqdm(dataset, desc="Running inference"), batch_size):
        batch = batch[0]
        predict = None
        response = ''
        while predict is None:
            response = model.generate(img_paths=batch['img_paths'], prompt=batch['prompt'],
                                      num_beams=num_beams, temperature=temperature)
            predict_num += 1
            predict = extract_answer(response, options=options)
            if predict is None:
                print(f"\nLabel: {batch['label']}; Invalid response: {response}")

        answer_dict = {'label': batch['label'], 'prompt': batch['prompt'], 'img_paths': batch['img_paths'],
                       'predict': predict, 'response': response, 'sentence': batch['sentence'],
                       'evaluation_id': batch['evaluation_id']}
        predictions.append(answer_dict)

    acc = calculate_score(predictions)
    print(f"Accuracy: {acc} of {len(predictions)} samples by predicting {predict_num} times")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))


class SparklesEvalDataset(Dataset):
    def __init__(
            self,
            dataset_root,
            data_path=None,
            data_from=0,
            data_to=10,
            dialogue_turn=1,
    ):
        if not data_path:
            data_path = f"{dataset_root}/annotations/sparkles_evaluation_sparkleseval_annotations.json"
        self.image_dir = f"{dataset_root}/images"
        with open(data_path, 'r') as fr:
            self.data = json.load(fr)
        self.data = self.data[data_from:data_to]
        self.dialogue_turn = dialogue_turn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        dialogue = item['dialogue']
        if self.dialogue_turn == 1:
            prompt = dialogue[0]['content']
            label = dialogue[1]['content']
            images = dialogue[0]['images']
        else:
            prompt = f"{dialogue[0]['content']}###Assistant: {dialogue[1]['content']}###Human: {dialogue[2]['content']}"
            label = dialogue[3]['content']
            images= dialogue[0]['images'] + dialogue[2]['images']

        img_paths = [os.path.join(self.image_dir, f"{image['image_id']}.jpg") for image in images]
        return {
            "img_paths": img_paths,
            "prompt": prompt,
            "label": label,
            "dialogue": dialogue,
            "evaluation_id": item['evaluation_id'],
        }


def evaluate_sparkles(model, dataset, answer_path, batch_size=1, dialogue_turn=1, num_beams=1, temperature=1.0):
    predictions = []
    for batch in more_itertools.chunked(tqdm(dataset, desc="Running inference"), batch_size):
        batch = batch[0]
        response = model.generate(img_paths=batch['img_paths'], prompt=batch['prompt'],
                                  num_beams=num_beams, temperature=temperature)
        predict = copy.deepcopy(batch['dialogue'])
        predict[dialogue_turn*2-1]['content'] = response
        predictions.append({
            'dialogue': predict,
            'evaluation_id': batch['evaluation_id'],
        })
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    return predictions


def extract_scores(s, criterion="C1"):
    # Split the string into evaluations
    evaluations = re.split(r'Evaluating [A-Za-z0-9]+', s)

    scores = []
    for evaluation in evaluations:
        # Find the first rating after a '(criterion)' in each evaluation
        match = re.search(fr'\({criterion}\)(.*?)Rating: \[?\[?(\d+\.?\d*)\]?\]?', evaluation, re.DOTALL)
        if match:
            scores.append(float(match.group(2)))
    return scores


def extract_all_scores(response):
    # extract the scores, A1_score extract from strings like "rating of A1 is [[8.67]]" -> "8.67"
    A1_score = re.findall(r"rating of A1 is \[?\[?(\d+\.?\d*)\]?\]?", response)
    A2_score = re.findall(r"rating of A2 is \[?\[?(\d+\.?\d*)\]?\]?", response)
    # if didn't find the score, output the response
    if len(A1_score) == 0 or len(A2_score) == 0:
        print(f"\nAttention: Didn't find the A1_score/A2_score, the response is {response}")
        return None, None

    # "(C1) ...... Rating: [[8]]" -> "8"
    C1_score = extract_scores(response, criterion="C1")
    C2_score = extract_scores(response, criterion="C2")
    C3_score = extract_scores(response, criterion="C3")
    if len(C1_score) != 2 or len(C2_score) != 2 or len(C3_score) != 2:
        print(f"\nAttention: Didn't find the C1_score/C2_score/C3_score, the response is {response}")
        return None, None

    A1_C123_scores = [float(A1_score[0]), float(C1_score[0]), float(C2_score[0]), float(C3_score[0])]
    A2_C123_scores = [float(A2_score[0]), float(C1_score[1]), float(C2_score[1]), float(C3_score[1])]
    # print(f"\nA1_C123_scores: {A1_C123_scores}; A2_C123_scores: {A2_C123_scores}")
    if int(np.mean(A1_C123_scores[1:])) != int(A1_C123_scores[0]):
        print(f"\nA1 score not match!")
    if int(np.mean(A2_C123_scores[1:])) != int(A2_C123_scores[0]):
        print(f"\nA2 score not match!")
    return A1_C123_scores, A2_C123_scores


def llm_as_judge(path, judge_model="gpt-3.5-turbo", data=None):
    judge_template = """Users will interact with a conversational assistant. The assistant is designed to understand, analyze, and reason about multiple images across two turns of conversation. The assistant is expected to provide highly helpful and exceptionally detailed answers providing comprehensive reasoning regarding the visual content of the images.

Below are images represented by their image ids and captions (delimited by triple quotes):
```json
{images}
```

Next is a dialogue between a user and the assistant regarding the images above:
```
###User Q1:
{Q1}

###Assistant A1:
{A1}

###User Q2:
{Q2}

###Assistant A2:
{A2}
```

Your task as an impartial judge is to evaluate the responses (A1 and A2) provided by the assistant to the user's questions.
Please rate the following three criteria C1, C2, and C3 on a scale of 1-10 for A1 and A2 separately, where a higher score indicates better overall performance:
(C1) Image Understanding and Reasoning: This measures the assistant's ability to accurately identify and describe objects, context, and relationships within and between the images.
(C2) Cross-Image and Cross-Turn Coherence: This evaluates the assistant's ability to maintain a consistent understanding across multiple images and dialogue turns.
(C3) Relevance and Completeness of Responses: This assesses whether the assistant's responses are directly related to the user's inquiries and the images' content, and whether the responses provide thorough, detailed answers.

Begin your evaluation by providing a short explanation for each criterion. Be as objective as possible. After providing your explanation, rate the response on a scale of 1 to 10 by strictly following the format below (note that "5" and "..." are placeholders):
```
* Evaluating A1
- (C1) Explanation: "..." Rating: [[5]]
- (C2) Explanation: "..." Rating: [[5]]
- (C3) Explanation: "..." Rating: [[5]]
Therefore, the overall rating of A1 is [[5]]

* Evaluating A2
- (C1) Explanation: "..." Rating: [[5]]
- (C2) Explanation: "..." Rating: [[5]]
- (C3) Explanation: "..." Rating: [[5]]
Therefore, the overall rating of A2 is [[5]]
```
"""

    if data is None:
        assert os.path.exists(path), f"File {path} not exists!"
        with open(path, 'r') as fr:
            data = json.load(fr)

    for idx, item in tqdm(enumerate(data), desc=f"Judging by {judge_model}"):
        dialogue = item['dialogue']
        Q1 = dialogue[0]['content'].replace("<Img><ImageHere></Img>", "")
        A1 = dialogue[1]['content']
        Q2 = dialogue[2]['content'].replace("<Img><ImageHere></Img>", "")
        A2 = dialogue[3]['content']
        images = dialogue[0]['images'] + dialogue[2]['images']
        prompt = judge_template.format(images=images, Q1=Q1, A1=A1, Q2=Q2, A2=A2)
        response = A1_C123_scores = A2_C123_scores = None
        while A1_C123_scores is None or A2_C123_scores is None:
            response = openai_chat_create(prompt, model=judge_model, max_tokens=1024)
            A1_C123_scores, A2_C123_scores = extract_all_scores(response)
        item[judge_model] = {"A1_C123_scores": A1_C123_scores, "A2_C123_scores": A2_C123_scores,
                     "prompt": prompt, "response": response,}

    A1_C123_scores_mean = np.mean([item[judge_model]['A1_C123_scores'] for item in data], axis=0)
    A2_C123_scores_mean = np.mean([item[judge_model]['A2_C123_scores'] for item in data], axis=0)
    print(f"A1_C123_scores_mean: {A1_C123_scores_mean}; A2_C123_scores_mean: {A2_C123_scores_mean}")

    with open(path, 'w') as fw:
        json.dump(data, fw, indent=4)

    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    # set --cfg-path that can be None
    parser.add_argument("--cfg-path", type=str, default=None, help="path to config file."
                                                                   "When set to None, evaluate gpt-4 annotations.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--sparkles_root", type=str, default='/path/to/Sparkles/', help="data root")
    parser.add_argument("--dataset", type=str, default='NLVR2', help="specify the dataset.")
    parser.add_argument("--seed", type=int, default=2023, help="")
    parser.add_argument("--data-from", type=int, default=0, help="specify the start idx of data.")
    parser.add_argument("--data-to", type=int, default=150, help="specify the end idx of data.")
    parser.add_argument("--merge-results", action="store_true", help="merge results from different runs.")
    parser.add_argument("--inference", action="store_true", help="inference mode.")
    parser.add_argument("--num-beams", type=int, default=1, help="num beams for beam search.")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling.")
    parser.add_argument('--judge_models', nargs='+', default=['gpt-4'], choices=['gpt-4', 'gpt-3.5-turbo'],
                        help='List of judge models. This argument only works when dataset is SparklesEval.')
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

    data_from = args.data_from
    data_to = args.data_to
    num_beams = args.num_beams
    temperature = args.temperature
    dataset_root = f"{args.sparkles_root}/evaluation/{args.dataset}"
    judge_models = args.judge_models

    if args.cfg_path is not None:
        cfg = Config(args)
        ckpt_path = cfg.model_cfg.ckpt
        beam = f"_beam{num_beams}" if num_beams > 1 else ""
        model_name = f"{ckpt_path.split('/')[-3]}_{ckpt_path.split('/')[-2]}_{ckpt_path.split('/')[-1][:-4]}{beam}"
        results_dir = f"{dataset_root}/results"
        results_path_name = f"{results_dir}/{args.dataset}_{model_name}"
    else:
        cfg = None
        model_name = "gpt-4"
        results_dir = f"{dataset_root}/annotations"
        results_path_name = f"{results_dir}/sparkles_evaluation_sparkleseval_annotations"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if args.inference:
        model = None
        if cfg and not (args.dataset == 'SparklesEval' and os.path.exists(f"{results_path_name}.json")):
            model = MiniGPT4(cfg, [args.gpu_id])
        answer_path = f"{results_path_name}_{data_from}to{data_to}.json"
        if not os.path.exists(os.path.dirname(answer_path)):
            os.makedirs(os.path.dirname(answer_path))
        if args.dataset == 'NLVR2':
            dataset = NLVR2Dataset(dataset_root=dataset_root, data_from=data_from, data_to=data_to)
            evaluate_pairs(model, dataset, answer_path=answer_path, options=dataset.options,
                           num_beams=num_beams, temperature=temperature)
        elif args.dataset == 'BISON':
            dataset = BISONDataset(dataset_root=dataset_root, data_from=data_from, data_to=data_to)
            evaluate_pairs(model, dataset, answer_path=answer_path, options=dataset.options,
                           num_beams=num_beams, temperature=temperature)
        elif args.dataset == 'SparklesEval':
            if os.path.exists(f"{results_path_name}.json"):
                # load the same generated dialogues for different judge models to evaluate
                data = json.load(open(f"{results_path_name}.json", 'r'))
                data = sorted(data, key=lambda x: (int(x['evaluation_id'])))
                data = data[data_from:data_to]
            else:
                dataset = SparklesEvalDataset(dataset_root=dataset_root, dialogue_turn=1, data_path=None,
                                              data_from=data_from, data_to=data_to)
                evaluate_sparkles(model, dataset, answer_path=answer_path, dialogue_turn=1,
                                  num_beams=num_beams, temperature=temperature)

                dataset = SparklesEvalDataset(dataset_root=dataset_root, dialogue_turn=2, data_path=answer_path,
                                              data_from=data_from-data_from, data_to=data_to-data_from)
                data = evaluate_sparkles(model, dataset, answer_path=answer_path, dialogue_turn=2,
                                  num_beams=num_beams, temperature=temperature)

            for judge_model in judge_models:
                data = llm_as_judge(answer_path, judge_model=judge_model, data=data)

        else:
            raise NotImplementedError

    if args.merge_results:
        # find answer files under dir(answer_path) and merge them into one answer file
        answer_path = f"{results_path_name}.json"
        answer_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir)
                        if f.startswith(answer_path.split('/')[-1][:-5] + "_") and f.endswith('.json')]
        answer_files = list(set(answer_files))
        predictions = []
        for answer_file in answer_files:
            with open(answer_file, 'r') as f:
                predictions.extend(json.load(f))
        predictions = sorted(predictions, key=lambda x: (int(x['evaluation_id'])))

        if args.dataset == 'SparklesEval':
            # merge the same generated dialogues for different judge models to evaluate
            predictions_dict = {}
            for prediction in predictions:
                if prediction['evaluation_id'] in predictions_dict:
                    for judge_model in judge_models:
                        if judge_model in prediction and judge_model not in predictions_dict[prediction['evaluation_id']]:
                            predictions_dict[prediction['evaluation_id']][judge_model] = prediction[judge_model]
                else:
                    predictions_dict[prediction['evaluation_id']] = prediction
            predictions = list(predictions_dict.values())

            html_save_path = f"{results_path_name}.html"
            visualize_dialogues_in_html(predictions, prediction_model=model_name,
                                        output_file=html_save_path, judge_models=judge_models)
            for judge_model in judge_models:
                if judge_model in predictions[0]:
                    A1_C123_scores_mean = np.mean([item[judge_model]['A1_C123_scores'] for item in predictions], axis=0)
                    A2_C123_scores_mean = np.mean([item[judge_model]['A2_C123_scores'] for item in predictions], axis=0)
                    print(f"{judge_model} as judge: A1_C123_scores_mean: {A1_C123_scores_mean}; "
                          f"A2_C123_scores_mean: {A2_C123_scores_mean}")
                    # calculate the mean of the last three items of A1_C123_scores_mean and A2_C123_scores_mean
                    A1 = np.mean(A1_C123_scores_mean[1:])
                    A2 = np.mean(A2_C123_scores_mean[1:])
                    overall = (A1 + A2) / 2
                    print(f"\n{judge_model} as judge: A1: {A1}; A2: {A2}; overall: {overall}")
        else:
            acc = calculate_score(predictions)
            print(f"{args.dataset} Accuracy: {acc:.3f}")
            answer_path = f"{results_path_name}_acc{acc:.3f}.json"
            html_save_path = f"{results_path_name}_acc{acc:.3f}.html"
            visualize_VLtasks_in_html(predictions, output_file=html_save_path)

        with open(answer_path, "w") as f:
            f.write(json.dumps(predictions, indent=4))
        # remove all answer files
        for answer_file in answer_files:
            os.remove(answer_file)
        print(f"Answer file saved to {answer_path}")