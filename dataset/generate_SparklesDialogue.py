import random
import traceback
from collections import Counter
import numpy as np
import re
import json
from tqdm.auto import tqdm
import os
import argparse
from call_gpt_api import openai_chat_create
from dataset.config_path import get_Sparkles_path


def append_image_tag(parsed_message, image_tag="<Img><ImageHere></Img>"):
    # simple replacement of "IMAGE#image_ids_1" with "IMAGE#image_ids_1<Img><ImageHere></Img>" is not enough,
    # because two image ids may have the same prefix, e.g. "IMAGE#1417" and "IMAGE#14"
    # which can result in twice matching of the same prefix
    # example:
    # 'image_ids': [804, 1417, 14],
    # 'content': 'Is there a potential narrative or story that connects the fire truck from IMAGE#804, the industrial machine in IMAGE#1417, and the man in a suit in IMAGE#14?'
    # after appending image tags:
    # wrong: 'Is there a potential narrative or story that connects the fire truck from IMAGE#804<Img><ImageHere></Img>, the industrial machine in IMAGE#14<Img><ImageHere></Img>17<Img><ImageHere></Img>, and the man in a suit in IMAGE#14?'
    # correct: 'Is there a potential narrative or story that connects the fire truck from IMAGE#804<Img><ImageHere></Img>, the industrial machine in IMAGE#1417<Img><ImageHere></Img>, and the man in a suit in IMAGE#14<Img><ImageHere></Img>?'
    for image_id in parsed_message["image_ids"]:
        # Create a regex pattern for the image_id to matche a string that either starts with 'IMAGE#'
        # or starts with '#' or is exactly the value of `image_id`, as long as `image_id` forms a complete word.
        #  The \b in the pattern ensures that we're matching whole words -- so #14 won't match #1417.
        # pattern = r"IMAGE#|#|{}\b".format(image_id)
        pattern = r"{}\b".format(image_id)
        # Create replacement string which will replace matched pattern
        replacement = "{}{}".format(image_id, image_tag)
        # Use re.sub with the IGNORECASE flag to match case-insensitive
        parsed_message["content"] = re.sub(pattern, replacement, parsed_message["content"], flags=re.IGNORECASE, count=1)


def check_image_tag(parsed_message, image_id_to_item, image_tag="<Img><ImageHere></Img>"):
    if parsed_message["content"].count(image_tag) != len(parsed_message["image_ids"]):
        print(f"Attention: The number of images does not match the number of {image_tag} in {parsed_message}!")
        return False
    if len(parsed_message["image_ids"]) > 1:
        # find the order of appearance of "image_ids_1<Img><ImageHere></Img>", "image_ids_2<Img><ImageHere></Img>"
        # in user's content, and then sort the image_ids in the same order
        # since the image features are embedded in the same order when training the model
        image_id_to_pos = {}
        for image_id in parsed_message["image_ids"]:
            pos = parsed_message["content"].find(f"{image_id}{image_tag}")
            if pos == -1:
                print("Attention: should not happen!")
                return False
            image_id_to_pos[image_id] = pos
        new_image_ids = [image_id for image_id, _ in sorted(image_id_to_pos.items(), key=lambda x: x[1])]
        if new_image_ids != parsed_message["image_ids"]:
            print(f"Warning: The order of images in {parsed_message} does not match the order of appearance in content!")
            parsed_message["image_ids"] = new_image_ids
    parsed_message["images"] = [image_id_to_item[str(image_id)] for image_id in parsed_message["image_ids"]]
    parsed_message.pop("image_ids", None)
    return True


def check_user_message(parsed_message, image_id_to_item, message_idx,
                       dialogue_idx=None, image_tag="<Img><ImageHere></Img>", start_img_num=1):
    if dialogue_idx and start_img_num and\
            (message_idx == 0 and len(parsed_message["image_ids"]) != dialogue_idx + start_img_num):
        # The number of images in 14/200 generated results do not follow the structured format
        # (almost all in the first message) but the results are still good, so we just ignore them
        num = len(parsed_message["image_ids"])
        print(f"Warning: message {message_idx} in dialogue {dialogue_idx} has {num} images!")
    if dialogue_idx and (message_idx == 2 and len(parsed_message["image_ids"]) != 1):
        # just a few (4/1700) cases generate two images instead of one in the second turn, filter them out
        num = len(parsed_message["image_ids"])
        print(f"Attention: message {message_idx} in dialogue {dialogue_idx} has {num} images!")
        return False

    for image_id in parsed_message["image_ids"]:
        if str(image_id) not in image_id_to_item and int(image_id) not in image_id_to_item:
            print(f"Attention: image_id ({image_id}) does not exist!")
            return False
    if image_tag not in parsed_message["content"]:
        append_image_tag(parsed_message)

    return check_image_tag(parsed_message, image_id_to_item)


def check_dialogue(parsed_dialogue, image_id_to_item, dialogue_idx=None, start_img_num=1):
    role2idx = {"user": [0, 2], "assistant": [1, 3]}
    if len(parsed_dialogue) != 4:
        print(f"Attention: The number of messages in each dialogue is {len(parsed_dialogue)} != 4")
        return False
    for message_idx, parsed_message in enumerate(parsed_dialogue):
        if parsed_message["role"] not in role2idx:
            print(f"Attention: The role of message {message_idx} is {parsed_message['role']}!")
            return False
        if parsed_message["role"] == "user" and "image_ids" in parsed_message:
            if not check_user_message(parsed_message, image_id_to_item, message_idx, dialogue_idx, start_img_num=start_img_num):
                return False
    return True


def try_parse_json(my_string):
    try:
        data = json.loads(my_string)
        return data
    except json.JSONDecodeError:
        return None


def parse_response_to_dialogue(response, image_id_to_item, dialogue_num=3):
    # parse the response whose format is like ("..." are illustrations):
    '''
    [
        [
            {'role': 'user', 'image_ids': [...], 'content': '...'},
            {'role': 'assistant', 'content': '...'},
            {'role': 'user', 'image_ids': [...], 'content': '...'},
            {'role': 'assistant', 'content': '...'}
        ],
        [
            {'role': 'user', 'image_ids': [...], 'content': '...'},
            {'role': 'assistant', 'content': '...'},
            {'role': 'user', 'image_ids': [...], 'content': '...'},
            {'role': 'assistant', 'content': '...'}
        ],
        [
            {'role': 'user', 'image_ids': [...], 'content': '...'},
            {'role': 'assistant', 'content': '...'},
            {'role': 'user', 'image_ids': [...], 'content': '...'},
            {'role': 'assistant', 'content': '...'}
        ]
    ]
    '''
    parsed_dialogues = []
    problem_dialogues = []
    # parse responses using regex since the response is not exactly follow the json format
    # and cannot directly parse it with json.loads(response)
    # first, split the response into three dialogues, each dialogue should start with "[{" and end with "}]"
    if dialogue_num == 3:
        start_img_num = 1
    elif dialogue_num == 2:
        start_img_num = 2
    else:
        start_img_num = None

    gen_dialogues = re.findall(r"\[\{.*?\}\]", response, re.DOTALL)
    response_string = response.replace("```json", "").replace("```", "")
    load_dialogues = try_parse_json(response_string)
    # if len(gen_dialogues) == 0:
    if load_dialogues:
        # possibly the response contains a lot of "\n" and spaces, and can be parsed by json.loads(response)
        # a few `Failed to parse the response with error` due to invalid or incomplete json (hasn't finished)
        # load_dialogues = json.loads(response_string)
        if len(load_dialogues) == 4 and type(load_dialogues[0]) is dict:  # may be only one dialogue with four messages
            load_dialogues = [load_dialogues]
        for dialogue_idx, load_dialogue in enumerate(load_dialogues):
            if check_dialogue(load_dialogue, image_id_to_item, dialogue_idx, start_img_num=start_img_num):
                parsed_dialogues.append(load_dialogue)
            else:
                problem_dialogues.append(load_dialogue)
        return parsed_dialogues, problem_dialogues
    if len(gen_dialogues) == 0 and dialogue_num == 1:
        gen_dialogues = [response_string]
    if len(gen_dialogues) != dialogue_num:
        # most len(gen_dialogues) == 2 due to the last dialogue is not finished because of the max_tokens limit
        print(f"Warning: The number of extracted dialogues is {len(gen_dialogues)} != {dialogue_num}")

    # second, parse each dialogue into four messages, each message should start with "{" and end with "}"
    for dialogue_idx, gen_dialogue in enumerate(gen_dialogues):
        flag = True
        parsed_dialogue = []
        gen_messages = re.findall(r"\{.*?\}", gen_dialogue, re.DOTALL)  # re.DOTALL means "." can match "\n"
        if len(gen_messages) != 4:
            print(f"Attention: The number of extracted messages is {len(gen_messages)} != 4")
            flag = False
        # third, parse each message into three fields: "role", "image_ids", "content"
        for message_idx, gen_message in enumerate(gen_messages):
            parsed_message = {}
            # find the "content" field between 'content': and the last "}" and remove the redundant " and '
            if "'content':" not in gen_message and "'caption':" in gen_message:
                # there are 25/200 cases generate "caption" instead of "content" in one of the messages
                # (usually in the third message, and mostly in the second dialogue)
                # the results are still good, so we just replace "caption" with "content"
                gen_message = gen_message.replace("'caption':", "'content':")
            if "'content':" not in gen_message and '"content":' in gen_message:
                gen_message = gen_message.replace('"content":', "'content':")
            find_content = re.findall(r"'content':(.*?)\}", gen_message, re.DOTALL)
            if len(find_content) == 0:
                flag = False
                print(f"Attention: message {message_idx} ({gen_message}) in dialogue {dialogue_idx} does not have content!")
                break
            # sometimes the "role" filed can be after the "content" field, so we need to remove it
            role_strings = ["'role': 'assistant'", "'role':'assistant'", '"role": "assistant"', '"role":"assistant"',
                            "'role': 'user'", "'role':'user'", '"role": "user"', '"role":"user"']
            for role_string in role_strings:
                if role_string in find_content[0]:
                    find_content[0] = find_content[0].replace(role_string, "")
            parsed_message["content"] = find_content[0].strip().rstrip(",").strip().strip('"').strip("'")
            if "'image_ids'" in gen_message or '"image_ids"' in gen_message:
                assert message_idx in [0, 2], "The message index of user role should be 0 or 2!"
                parsed_message["role"] = "user"
                # the list of image_ids should start with "[" and end with "]"
                parsed_message["image_ids"] = [_.strip().strip("'") for _ in re.findall(r"\[(.*?)\]", gen_message)[0].split(',')]
                if not check_user_message(parsed_message, image_id_to_item, message_idx, dialogue_idx, start_img_num=start_img_num):
                    flag = False
                    break
            else:
                assert message_idx in [1, 3], "The message index of assistant role should be 1 or 3!"
                parsed_message["role"] = "assistant"
            parsed_dialogue.append(parsed_message)
        if len(parsed_dialogue) != 4:
            flag = False
            print(f"Attention: The number of messages in each dialogue is {len(parsed_dialogue)} != 4")
        if flag:
            parsed_dialogues.append(parsed_dialogue)
        else:
            problem_dialogues.append(parsed_dialogue)
    if len(parsed_dialogues) != dialogue_num:
        print(f"Warning: The number of parsed dialogues is {len(parsed_dialogues)} != {dialogue_num}")
    return parsed_dialogues, problem_dialogues


def parse_responses_to_dialogues(raw_data, image_id_to_item, dialogue_num=3):
    parsed_results = []
    problem_results = []
    for i, item in enumerate(raw_data):
        # print(f"\n----------------Case {i}------------------\n")
        prompt = item["raw"][0]["prompt"]
        response = item["raw"][0]["response"]
        try:
            parsed_dialogues, problem_dialogues = parse_response_to_dialogue(response, image_id_to_item, dialogue_num)
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to parse the response with error: {e}")
            parsed_dialogues = None
            problem_dialogues = None
        raw = [{"prompt": prompt, "response": response}]
        parsed_results.append({"dialogues": parsed_dialogues, "raw": raw})
        problem_results.append({"dialogues": problem_dialogues, "raw": raw})

    return parsed_results, problem_results


def format_SparklesDialogue(parsed_results, save_basepath, include_turn1=True, include_turn2=True):
    template1 = "###Human: {question}###Assistant: "
    template2 = "###Human: {question}###Assistant: {answer}###Human: {question2}###Assistant: "
    results_to_save = [[], [], [], []]
    for item in parsed_results:
        dialogue = item["dialogue"]
        instruction1 = template1.format(question=dialogue[0]["content"])
        instruction2 = template2.format(question=dialogue[0]["content"], answer=dialogue[1]["content"],
                                        question2=dialogue[2]["content"])
        if include_turn1:
            turn1 = {"uniform_question": dialogue[0]["content"], "instruction": instruction1,
                      "answer": dialogue[1]["content"], "tgt_imgs": dialogue[0]["images"]}
            results_to_save[len(turn1["tgt_imgs"]) - 1].append(turn1)
        if include_turn2:
            turn2 = {"uniform_question": dialogue[2]["content"], "instruction": instruction2,
                               "answer": dialogue[3]["content"], "tgt_imgs": dialogue[0]["images"] + dialogue[2]["images"]}
            results_to_save[len(turn2["tgt_imgs"]) - 1].append(turn2)

    for img_num, result in enumerate(results_to_save):
        if len(result) == 0:
            continue
        if include_turn1 and include_turn2:
            save_name = f"{save_basepath}_{img_num + 1}img.json"
        elif include_turn1:
            save_name = f"{save_basepath}_turn1_{img_num + 1}img.json"
        elif include_turn2:
            save_name = f"{save_basepath}_turn2_{img_num + 1}img.json"
        else:
            raise ValueError("At least one of include_turn1 and include_turn2 should be True!")
        with open(save_name, "w") as f:
            json.dump(result, f)

    # return results_to_save


def parse_dialogue_data(image_id_to_item, results_path, raw_data=None, raw_data_paths=None, dialogue_num=3,
                        demo_dialogues_list=None, include_turn1=True, include_turn2=True):
    if raw_data is None:
        assert raw_data_paths is not None, "Either raw_data or raw_data_paths should be provided!"
        raw_data = []
        for raw_data_path in raw_data_paths:
            with open(raw_data_path, "r") as f:
                raw_data.extend(json.load(f))

    results, problem_results = parse_responses_to_dialogues(raw_data, image_id_to_item, dialogue_num)
    # construct data for dataloader, categorize by the number of images, each data in the format of
    # [{"answer": answer, "uniform_question": question, "tgt_imgs": tgt_imgs})]
    parsed_results = []
    for idx, item in enumerate(results):
        if item["dialogues"] is None or len(item["dialogues"]) == 0:
            continue
        for dialogue in item["dialogues"]:
            parsed_results.append({"dialogue": dialogue, "raw": item["raw"]})

    if demo_dialogues_list is not None:
        all_demo_dialogues = []
        for demo_dialogues in demo_dialogues_list:
            all_demo_dialogues.extend(demo_dialogues)

        for idx, dialogue in enumerate(all_demo_dialogues):
            if not check_dialogue(dialogue['dialogue'], image_id_to_item):
                all_demo_dialogues.pop(idx)

        parsed_results.extend(all_demo_dialogues)

    format_SparklesDialogue(parsed_results, save_basepath=results_path[:-5],
                            include_turn1=include_turn1, include_turn2=include_turn2)

    with open(results_path, "w") as f:
        json.dump(parsed_results, f)

    return parsed_results


def prompt_SparklesDialogueVG(image_id_to_item, raw_data_path, demo_dialogues_list,
                                        case_num=5, candidate_image_num=4, numstr="two"):
    template = """Users will interact with a conversational assistant that has advanced capabilities of understanding, analyzing, and reasoning about images. This includes discussing a variety of real-world concepts, objects, and entities, generating a range of text materials, seeking advice, guidance, or assistance, and much more.

Below are an illustrative dialogue presented in a JSON format. The dialogue represents a meaningful conversation between a "user" and the "assistant" regarding multiple images. Each "user" message contains an "image_ids" field recording the IDs of newly selected images. The images are referred to in the "content" field as IMAGE#image_id.
```json
{}
```
Please note that the user contents in the JSON above may be a counterexample that reveals the content of images and can be answered without looking at the images. Please make sure not to reveal the content of the images or describe the images in the user messages in the conversation that follows.

Please note that the specific "image_ids" and "content" in the JSON above are for illustrative purposes only. The actual candidate images are shown below delimited by triple quotes, each accompanied by an image ID and a caption. Avoid using phrases similar to 'caption' and 'description' in your dialogue as if the user and the assistant have visual capabilities.
```json
{}
```
Each dialogue consists of four messages:
1. A user examines all candidate images, selects {} highly relevant images, and sends a reasonable and creative message to the assistant.
2. Once the images are provided, the assistant thoroughly perceives and comprehends them, responding with highly helpful and exceptionally detailed answers that provide comprehensive reasoning regarding the visual content of the images.
3. Considering the past dialogue, the user chooses other candidate images for further inquiry. The user should refer to both the newly selected images and those mentioned earlier in the same dialogue.
4. The assistant provides a highly helpful and exceptionally detailed answer providing comprehensive reasoning regarding the visual content of the images.

The following is a dialogue between the user and the assistant, adhering to the given JSON format.
Make sure to formulate accurate and diverse "content" that does not follow the illustrative dialogues. And remember to develop the last "content" even though it is shown as "..." in the JSON format provided above."""

    # get all image ids used in the demo dialogues
    demo_image_ids = []
    for demo_dialogues in demo_dialogues_list:
        for demo_dialogue in demo_dialogues:
            for demo_message in demo_dialogue["dialogue"]:
                if "images" in demo_message:
                    demo_image_ids.extend([int(image["image_id"]) for image in demo_message["images"]])
    demo_image_ids = list(set(demo_image_ids))
    # exclude the demo_image_ids from image_id_to_item
    image_id_to_item = {k: v for k, v in image_id_to_item.items() if k not in demo_image_ids}
    # remove top 100 items as their image ids may be used in real testing dialogues
    image_id_to_item = {k: v for k, v in image_id_to_item.items() if int(k) > 100}

    # we need to fill the demo dialogues and some image candidates in the template
    # read the demo dialogues
    raw_data = []
    for i in tqdm(range(case_num)):
        # randomly sample demo dialogues
        demo_dialogues = [random.choice(_)["dialogue"] for _ in demo_dialogues_list]
        demo_dialogues_short = demo_dialogues
        for demo_dialogue in demo_dialogues_short:
            for message_idx, message in enumerate(demo_dialogue):
                if "images" in message:
                    # for the user role, extract the 'image_id' from 'images' field, save 'image_id' instead of 'images'
                    # convert to int to be consistent with SVIT (VG) image ids
                    message["image_ids"] = [int(image["image_id"]) for image in message["images"]]
                    message.pop("images")
                    message["content"] = message["content"].replace("<Img><ImageHere></Img>", "")
                    # make message["image_ids"] shown before message["content"] when str(message)
                    # to imitate the process of firstly selecting images and then writing the message
                    message["content"] = message.pop("content")
                    # message = {k: message[k] for k in ["role", "image_ids", "content"]}
                if message["role"] == "assistant" and message_idx == 3:
                    # keep the content of the first turn as context, remove the second turn
                    message["content"] = "..."  # to shorten the demo dialogue and reduce noise

        candidates = random.sample(list(image_id_to_item.values()), candidate_image_num)
        # remove sampled candidates from image_id_to_item to increase diversity
        for candidate in candidates:
            image_id_to_item.pop(str(candidate["image_id"]))
        prompt = template.format(demo_dialogues_short, candidates, numstr)
        print(f"\n----------------Case {i}------------------\n")
        print(prompt)
        response = openai_chat_create(prompt)
        print(response)

        raw_data.append({"raw": [{"prompt": prompt, "response": response}]})

        if (i + 1) % 20 == 0:
            save_path = raw_data_path.replace(".json", "_i{}.json".format(i))
            with open(save_path, 'w') as f:
                json.dump(raw_data, f)

    with open(raw_data_path, 'w') as f:
        json.dump(raw_data, f)
    return raw_data


def prompt_SparklesDialogueCC(image_id_to_item, raw_data_path, demo_dialogues_list, case_num=5, candidate_image_num=9):
    template = """Users will interact with a conversational assistant that has advanced capabilities of understanding, analyzing, and reasoning about images. This includes discussing a variety of real-world concepts, objects, and entities, generating a range of text materials, seeking advice, guidance, or assistance, and much more.

Below are three illustrative dialogues presented in a JSON format. Each one represents a self-contained conversation between a "user" and the "assistant" regarding multiple images. Each "user" message contains an "image_ids" field recording the IDs of newly selected images. The images are referred to in the "content" field as IMAGE#image_id.
```json
{}
```

Please note that the specific "image_ids" and "content" in the JSON above are for illustrative purposes only. The actual candidate images are shown below delimited by triple quotes, each accompanied by an image ID and a caption. Avoid using phrases similar to 'caption' and 'description' in your dialogue as if the user and the assistant have visual capabilities.
```json
{}
```

Each dialogue consists of four messages:
1. A user examines all candidate images, selects highly relevant ones, and sends a reasonable and creative message to the assistant.
2. Once the images are provided, the assistant thoroughly perceives and comprehends them, responding with highly helpful and exceptionally detailed answers that provide comprehensive reasoning.
3. Considering the past dialogue, the user chooses another candidate image for further inquiry. The user should refer to both the newly selected image and those mentioned earlier in the same dialogue.
4. The assistant provides a highly helpful and exceptionally detailed answer providing comprehensive reasoning regarding the visual content of the images.

The following are three independent dialogues between the user and the assistant, adhering to the given JSON format. In this format, the first message in the three dialogues includes 1, 2, and 3 image ids respectively.
Make sure to formulate accurate and diverse "content" that does not strictly follow the illustrative dialogues. And remember to develop the last "content" even though it is shown as "..." in the JSON format provided above."""

    # get all image ids used in the demo dialogues
    demo_image_ids = []
    for demo_dialogues in demo_dialogues_list:
        for demo_dialogue in demo_dialogues:
            for demo_message in demo_dialogue["dialogue"]:
                if "images" in demo_message:
                    demo_image_ids.extend([image["image_id"] for image in demo_message["images"]])
    demo_image_ids = list(set(demo_image_ids))
    # exclude the demo_image_ids from image_id_to_item
    image_id_to_item = {k: v for k, v in image_id_to_item.items() if k not in demo_image_ids}

    # we need to fill the demo dialogues and some image candidates in the template
    # read the demo dialogues
    raw_data = []
    for i in tqdm(range(case_num)):
        # randomly sample a demo dialogue
        demo_dialogues = [random.choice(_)["dialogue"] for _ in demo_dialogues_list]
        demo_dialogues_short = demo_dialogues
        for demo_dialogue in demo_dialogues_short:
            for message_idx, message in enumerate(demo_dialogue):
                if "images" in message:
                    # for the user role, extract the 'image_id' from 'images' field, save 'image_id' instead of 'images'
                    message["image_ids"] = [image["image_id"] for image in message["images"]]
                    message.pop("images")
                    message["content"] = message["content"].replace("<Img><ImageHere></Img>", "")
                    # make message["image_ids"] shown before message["content"] when str(message)
                    # to imitate the process of firstly selecting images and then writing the message
                    message["content"] = message.pop("content")
                    # message = {k: message[k] for k in ["role", "image_ids", "content"]}
                if message["role"] == "assistant" and message_idx == 3:
                    # keep the content of the first turn as context, remove the second turn
                    message["content"] = "..."  # to shorten the demo dialogue and reduce noise

        candidates = random.sample(list(image_id_to_item.values()), candidate_image_num)
        prompt = template.format(demo_dialogues_short, candidates)
        print(f"\n----------------Case {i}------------------\n")
        print(prompt)
        response = openai_chat_create(prompt)
        print(response)

        raw_data.append({"raw": [{"prompt": prompt, "response": response}]})

        if (i + 1) % 20 == 0:
            save_path = raw_data_path.replace(".json", "_i{}.json".format(i))
            with open(save_path, 'w') as f:
                json.dump(raw_data, f)

    with open(raw_data_path, 'w') as f:
        json.dump(raw_data, f)
    return raw_data


def format_SVIT_description(data, save_path, remove_img_ids=None):
    if remove_img_ids is None:
        remove_img_ids = []
    results = []
    for item in data:
        assert len(item['conversations']) == 1
        question = item['conversations'][0]['content'][0]['value']
        answer = item['conversations'][0]['content'][1]['value']
        image_id = item['image_id']
        # Filter out those image ids in the Sparkles dataset
        if image_id in remove_img_ids:
            continue
        uniform_question = question.replace("the image", f"the IMAGE#{image_id}<Img><ImageHere></Img>")
        results.append({"question": question, "answer": answer, "uniform_question": uniform_question,
                        "tgt_imgs": [{'image_id': image_id, 'caption': answer}]})
    with open(save_path, 'w') as f:
        json.dump(results, f)
    return results


def format_LLaVA(data, save_path, remove_img_ids=None):
    if remove_img_ids is None:
        remove_img_ids = []
    results = []
    image_placeholders = ["<Img><ImageHere></Img>", "image <Img><ImageHere></Img>",
                   "IMAGE#{image_id}<Img><ImageHere></Img>", "image of IMAGE#{image_id}<Img><ImageHere></Img>"]
    tails = ["", "in", "in the", "based on this", "based on", "from the", "for", "regarding"]
    heads = ["", "In", "In the", "based on this", "Based on", "From the", "For", "Regarding",
             "Looking at", "Given", "Considering", "Seeing", "Analyze"]
    for item in data:  # 76643
        assert item['conversations'][0]['from'] == 'human'
        assert item['conversations'][1]['from'] == 'gpt'
        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        image_id = item['id']
        # Filter out those image ids in the Sparkles dataset
        if int(image_id) in remove_img_ids:
            continue
        question = question.replace("\n<image>", "").replace("<image>\n", "")
        image_placeholder = random.choice(image_placeholders).format(image_id=image_id)
        if "image" in question:  # 5892
            uniform_question = question.replace("image", image_placeholder, 1)
        else:
            if random.random() < 0.5:
                uniform_question = random.choice(heads) + " " + image_placeholder + ", " + question[0].lower() + question[1:]
            else:
                uniform_question = question[:-1] + " " + random.choice(tails) + " " + image_placeholder + question[-1]

        results.append({"question": question, "answer": answer, "uniform_question": uniform_question,
                        "tgt_imgs": [{'image_id': image_id, 'caption': answer}]})
    with open(save_path, 'w') as f:
        json.dump(results, f)
    return results


def get_demo_dialogues_VG(dialogues_path, SparklesDialogueCC, target_num_images, sparkles_root):
    with open(dialogues_path, 'r') as fr:
        dialogues = json.load(fr)

    from data_statistics import get_sentences, group_and_rank_questions_by_verb_noun
    save_name, sentences, dialogue_indices = get_sentences(SparklesDialogueCC, [0], [target_num_images], subset="CC")
    grouped_questions_ranked = group_and_rank_questions_by_verb_noun(sentences, dialogue_indices, sparkles_root, save_name)

    demo_dialogues = []
    for item in grouped_questions_ranked:
        if item['count'] > 1:
            continue
        dialogue = dialogues[item['instr_dialogue'][0]['dialogue_idx']]['dialogue']
        if len(dialogue[1]['content'].split()) < 100 or len(dialogue[3]['content'].split()) < 100:
            continue
        demo_dialogues.append({"dialogue": dialogue})
    demo_dialogues_list = [demo_dialogues]
    return demo_dialogues_list


def get_data_img_ids(BISON_path, vg_image_data_path, parsed_results):
    '''
    NLVR2 (google similar images), MiniGPT-4 (Conceptual Captions) should have no overlap with other data (COCO/VG);
    SparklesDialogueVG from SVIT, SVIT from VG, VG from COCO, BISON (cocoval2014) & LLaVA (cocotrain2017) from COCO,
    thus need to remove those overlapped images. ->
    First remove BISON eval from Sparkles Train.
    Then, remove all images in Sparkles (Train & Eval) & BISON eval from SVIT & LLaVA.
    # NLVR2 doesn't use Visual Genome images
    '''
    vg_img_ids = []
    for jdx, item in enumerate(parsed_results):
        dialogue = item["dialogue"]
        for msg in dialogue:
            if "images" in msg:
                vg_img_ids.extend([_["image_id"] for _ in msg["images"]])

    with open(vg_image_data_path, 'r') as fr:
        vg_image_data = json.load(fr)
    cocoid2vgid = {}
    vgid2cocoid = {}
    for item in vg_image_data:
        if item['coco_id']:
            cocoid2vgid[item['coco_id']] = item['image_id']
            vgid2cocoid[item['image_id']] = item['coco_id']

    coco_img_ids = []
    with open(BISON_path, 'r') as f:
        eval_data = json.load(f)
    for idx, item in enumerate(eval_data):
        for cand in item['image_candidates']:
            coco_img_ids.append(cand['image_id'])
            if cand['image_id'] in cocoid2vgid and cocoid2vgid[cand['image_id']] in vg_img_ids:
                print(idx, vg_img_ids.index(cocoid2vgid[cand['image_id']]))
                break
            elif cand['image_id'] in cocoid2vgid:
                vg_img_ids.append(cocoid2vgid[cand['image_id']])

    coco_img_ids.extend([vgid2cocoid[vg_id] for vg_id in vg_img_ids if vg_id in vgid2cocoid])
    coco_img_ids = list(set(coco_img_ids))
    return vg_img_ids, coco_img_ids


def get_image_text_pairs_CC(sparkles_root, demo_dialogues_list):
    filter_cap_path = os.path.join(sparkles_root, 'data', 'cc_sbu_align', 'filter_cap.json')
    with open(filter_cap_path, 'r') as f:
        filter_cap = json.load(f)
    image_id_to_item = {item['image_id']: item for item in filter_cap['annotations']}

    # remove image ids used in all demo_dialogues_list from image_id_to_item to reduce duplication
    image_ids = []
    for demo_dialogues in demo_dialogues_list:
        for i, d in enumerate(demo_dialogues):
            for turn in d['dialogue']:
                if "images" in turn:
                    image_ids.extend([_['image_id'] for _ in turn['images']])
    # count the number of times each image is used, and sort the images by their counts
    image_count = Counter(image_ids)
    sorted_image_count = sorted(image_count.items(), key=lambda x: x[1], reverse=True)
    for image_id, _ in sorted_image_count:
        image_id_to_item.pop(image_id)

    return image_id_to_item


def get_image_text_pairs_VG(sparkles_root):
    image_id_to_item = {}
    path = os.path.join(sparkles_root, 'data', 'SVIT', 'detail_description.json')
    with open(path, 'r') as f:
        svit_data = json.load(f)
    for item in svit_data:
        assert len(item['conversations']) == 1
        answer = item['conversations'][0]['content'][1]['value']
        image_id = item['image_id']
        image_id_to_item[str(image_id)] = {'image_id': image_id, 'caption': answer}
    return image_id_to_item



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--case_num", type=int, default=1220, help=(""))
    parser.add_argument("--seed", type=int, default=0, help=(""))
    parser.add_argument("--sparkles_root", type=str, default=None, help=(""))
    parser.add_argument("--vg_image_data_path", type=str, default="/path/to/VisualGenome/image_data.json", help=(""))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    sparkles_root, SparklesDialogueCC_root, SparklesDialogueVG_root, SparklesDialogueCC_path, \
    SparklesDialogueVG_path, SparklesEval_path, BISON_path, NLVR2_path, statistics_dir = \
        get_Sparkles_path(args.sparkles_root)
    # -------------------------------------------  SparklesDialogueCC -------------------------------------------

    demo_dialogues_list = []
    for target_image_num in range(1, 4):
        demo_dialogues_path = os.path.join(SparklesDialogueCC_root, "annotations",
                                           f'SparklesDialogueCC_50demo_img{target_image_num}1.json')
        if os.path.exists(demo_dialogues_path):
            with open(demo_dialogues_path, 'r') as fr:
                demo_dialogues_list.append(json.load(fr))

    image_text_pairs_CC = get_image_text_pairs_CC(sparkles_root, demo_dialogues_list)
    raw_data_path = os.path.join(SparklesDialogueCC_root, "annotations", "SparklesDialogueCC_raw.json")
    SparklesDialogueCC_raw = prompt_SparklesDialogueCC(
        image_text_pairs_CC, raw_data_path, demo_dialogues_list, case_num=args.case_num)
    SparklesDialogueCC = parse_dialogue_data(image_text_pairs_CC, SparklesDialogueCC_path, dialogue_num=3,
                                             raw_data=SparklesDialogueCC_raw, demo_dialogues_list=demo_dialogues_list)

    # -------------------------------------------  SparklesDialogueVG -------------------------------------------

    image_text_pairs_VG = get_image_text_pairs_VG(sparkles_root)

    SparklesDialogueVG_raw = []
    for target_image_num in range(2, 4):
        numstr = "two" if target_image_num == 2 else "three"
        demo_dialogues_list_VG = get_demo_dialogues_VG(
            SparklesDialogueCC_path, SparklesDialogueCC, target_image_num, sparkles_root)
        raw_data_path = os.path.join(SparklesDialogueVG_root, "annotations",
                                     f"SparklesDialogueVG_raw_{target_image_num}img.json")
        res = prompt_SparklesDialogueVG(image_text_pairs_VG, raw_data_path,
                                        demo_dialogues_list_VG, case_num=args.case_num, numstr=numstr)
        SparklesDialogueVG_raw.extend(res)

    SparklesDialogueVG = parse_dialogue_data(image_text_pairs_VG, SparklesDialogueVG_path,
                                             raw_data=SparklesDialogueVG_raw, dialogue_num=1)

    # -------------------------------------------  LLaVA & SVIT -------------------------------------------

    with open(SparklesEval_path, 'r') as fr:
        SparklesEval = json.load(fr)
    parsed_results = SparklesDialogueVG + SparklesEval
    vg_img_ids, coco_img_ids = get_data_img_ids(BISON_path, args.vg_image_data_path, parsed_results)
    paths = [os.path.join(sparkles_root, "data", "LLaVA", 'complex_reasoning_77k.json'),
             os.path.join(sparkles_root, "data", "LLaVA", 'detail_23k.json')]
    for path in paths:
        name = path.split("/")[-1].split(".")[0]
        save_path = os.path.join(sparkles_root, "data", "LLaVA", f"LLaVA_{name}_filtered_for_Sparkles.json")
        with open(path, 'r') as f:
            llava_data = json.load(f)
        format_LLaVA(llava_data, save_path, remove_img_ids=coco_img_ids)

    # path = os.path.join(sparkles_root, "data", "SVIT", 'SVIT/detail_description.json')
    # name = path.split("/")[-1].split(".")[0]
    # save_path = os.path.join(sparkles_root, "data", "SVIT", f"SVIT_{name}_filtered_for_Sparkles.json")
    # format_SVIT_description(desc_data, save_path, remove_img_ids=vg_img_ids)
