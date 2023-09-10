# ref: https://github.com/Vision-CAIR/MiniGPT-4/pull/232/files
import argparse
import random
from datetime import datetime
import pytz
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from sparkles.common.config import Config
from sparkles.common.dist_utils import get_rank
from sparkles.common.registry import registry
from sparkles.conversation.conversation_sparkleschat import Chat, CONV_VISION

# imports modules for registration
from sparkles.datasets.builders import *
from sparkles.models import *
from sparkles.processors import *
from sparkles.runners import *
from sparkles.tasks import *
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--save-root", type=str, default="/path/to/Sparkles/demo/",
                        help="root to saved dialogues")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================

def gradio_clear(chat_state, img_list, img_emb_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list.clear()
    if img_emb_list is not None:
        img_emb_list.clear()
    return chat_state, img_list, img_emb_list


def upload_img(gr_img, chat_state, img_list, img_emb_list):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, img_list, img_emb_list
    img_list.append(gr_img)
    # upload an image to the chat
    chat.upload_img(gr_img, chat_state, img_emb_list)
    # update image, text_input, upload_button, chat_state, gallery, img_emb_list
    return gr.update(value=None, interactive=True), \
           gr.update(interactive=True, placeholder='Type and press Enter. Use ⭐ to refer to the images in the same order as they were uploaded.'), \
           gr.update(value="Send more images", interactive=True), chat_state, img_list, img_emb_list


def gradio_ask(user_message, chatbot, chat_state, img_list):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    history_img_cnt = 0
    for turn in chatbot:
        for msg in turn:
            history_img_cnt += msg.count('✨')
            history_img_cnt += msg.count('⭐')
    if len(img_list) != history_img_cnt + user_message.count('✨') + user_message.count('⭐'):
        warning_label = f"Please insert a total of {len(img_list)} ⭐ in your messages (including history messages) to indicate the insertion of the uploaded images."
        return gr.update(interactive=True,
                         placeholder=f'Please insert {len(img_list) - history_img_cnt} ⭐ to indicate '
                                     f'the insertion of newly uploaded images.'), chatbot, chat_state, warning_label
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state, ''


def gradio_answer(warning_label, chatbot, chat_state, img_list, num_beams, temperature, max_new_tokens, gallery):
    temperature = float(temperature)
    if warning_label['label'] != '':
        return chatbot, chat_state, gr.update(interactive=True), gr.update(value="Send more image", interactive=True), warning_label['label']
    llm_message, _, warning_label = chat.answer(conv=chat_state,
                                              img_list=img_list,
                                              num_beams=num_beams,
                                              temperature=temperature,
                                              max_new_tokens=max_new_tokens,
                                              max_length=2000)
    chatbot[-1][1] = llm_message
    callback.flag([gallery, chatbot], flag_option="automatic")
    # update chatbot, chat_state, image, upload_button
    return chatbot, chat_state, gr.update(interactive=True), gr.update(value="Send more image", interactive=True), warning_label



title = """<h1 align="center">Chat with ✨Sparkles✨</h1>"""
article = """<h2>✨Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models.</h2>
<h3>For more details, check out our <a href='https://arxiv.org/pdf/2308.16463.pdf'>paper</a> and <a href='https://github.com/HYPJUDY/Sparkles'>code</a>! Feel free to contact <a href='https://hypjudy.github.io/website/'>Yupan Huang</a> for further inquiries.</h3>"""
usage = """<p><strong>To chat with Sparkles across multiple images, use ⭐ to denote each image within your prompt, just as you would with words.</strong> Label the images as IMAGE#1⭐, IMAGE#2⭐, and so on, or you can use phrases like "the second image" to refer to specific images.</p>
    <p><strong>💫Need inspiration?</strong> Why not explore the examples at the bottom of this page? Ask Sparkles to generate creative text, compare or connect different images, offer advice, and much more. Your prompts change the responses a lot. <strong>Have fun!</strong></p>
    <p>🌟Sparkles loves <strong>natural images</strong> but might struggle with text-rich or domain-specific images. Also, it may produce <strong>inaccurate information</strong> about details, objects, or facts.</p>
    """
# terms of use refer to: https://chat.lmsys.org/
terms = """<strong>🌠By using this service, you agree to these Terms of Use: </strong> The service is a research preview. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license."""

with gr.Blocks(allow_flagging="auto", title="✨Sparkles✨") as demo:
    gr.Markdown(title)
    gr.Markdown(article)
    gr.Markdown(usage)
    gr.Markdown(terms)

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", interactive=True)
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")

            warning_label = gr.Label("Please use ⭐ in your message to refer to the images "
                                     "in the same order as they were uploaded.")

            with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    interactive=True,
                    label="beam search numbers",
                )

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

                max_new_tokens = gr.Slider(
                    minimum=128,
                    maximum=1024,
                    value=768,
                    step=128,
                    interactive=True,
                    label="max_new_tokens",
                )

            with gr.Row(visible=True) as button_row:
                upvote_btn = gr.Button(value="👍Upvote")
                downvote_btn = gr.Button(value="👎Downvote")
                flag_btn = gr.Button(value="⚠Flag")
                clear = gr.ClearButton([image, warning_label])

        with gr.Column(scale=2):
            chat_state = gr.State(CONV_VISION.copy())
            img_list = gr.State([])
            img_emb_list = gr.State([])
            gallery = gr.Gallery(label="Images", show_label=True) \
                .style(rows=[2], object_fit="scale-down", height="400px", preview=True)
            chatbot = gr.Chatbot(label='Sparkles')
            text_input = gr.Textbox(label='Me', placeholder='Please upload your image first', interactive=False)
            clear.add([chatbot, text_input, gallery])

    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "images")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Examples(
                label="Example 1: images",
                examples=[
                    [f"{image_dir}/case1_1.jpg"], [f"{image_dir}/case1_2.jpg"],
                    [f"{image_dir}/case1_3.jpg"], [f"{image_dir}/case1_4.jpg"],
                ],
                inputs=[image],
            )
        with gr.Column(scale=3):
            gr.Examples(
                label="Example 1: text",
                examples=[
                    ["Create a story that takes place in ⭐ for the characters depicted in ⭐."],
                    ["Imagine a dialogue between Harry Potter and ⭐ that takes place in the scene of ⭐."],
                ],
                inputs=[text_input],
            )
    with gr.Row():
        with gr.Column(scale=2):
            gr.Examples(
                label="Example 2: images",
                examples=[
                    [f"{image_dir}/case2_1.jpg"], [f"{image_dir}/case2_2.jpg"], [f"{image_dir}/case2_3.jpg"],
                ],
                inputs=[image],
            )
        with gr.Column(scale=3):
            gr.Examples(
                label="Example 2: text",
                examples=[
                    ["Create a song where the scene twists from ⭐ to ⭐."],
                    ["Create a title for this song that takes inspiration from ⭐."],
                ],
                inputs=[text_input],
            )
    with gr.Row():
        with gr.Column(scale=2):
            gr.Examples(
                label="Example 3: images",
                examples=[
                    [f"{image_dir}/case3_1.jpg"], [f"{image_dir}/case3_2.jpg"], [f"{image_dir}/case3_3.jpg"],
                    [f"{image_dir}/case3_4.jpg"], [f"{image_dir}/case3_5.jpg"],
                ],
                inputs=[image],
            )
        with gr.Column(scale=3):
            gr.Examples(
                label="Example 3: text",
                examples=[
                    ["I'm showing my friends around a building. Its outside looks like ⭐ and the inside looks like ⭐. Please tell them about the building."],
                    ["Here is the exterior ⭐ and interior ⭐ of another building. Could you compare the similarities and differences between the two buildings?"],
                    ["What culture does ⭐ convey?"]
                ],
                inputs=[text_input],
            )

    callback = gr.CSVLogger()
    # This needs to be called at some point prior to the first call to callback.flag()
    current_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S")
    callback.setup([gallery, chatbot, temperature, max_new_tokens, upvote_btn, downvote_btn, flag_btn],
                   f"{args.save_root}/{current_time}")

    upload_button.click(upload_img, [image, chat_state, img_list, img_emb_list],
                        [image, text_input, upload_button, chat_state, gallery, img_emb_list])

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state, img_list],
                      [text_input, chatbot, chat_state, warning_label])\
        .then(gradio_answer,
              [warning_label, chatbot, chat_state, img_emb_list, num_beams, temperature, max_new_tokens, gallery],
              [chatbot, chat_state, image, upload_button, warning_label])\

    clear.click(gradio_clear, [chat_state, img_list, img_emb_list], [chat_state, img_list, img_emb_list])
    upvote_btn.click(lambda *args: callback.flag(args, flag_option="upvote"),
                     [gallery, chatbot, temperature, max_new_tokens, upvote_btn], None, preprocess=False)
    downvote_btn.click(lambda *args: callback.flag(args, flag_option="downvote"),
                       [gallery, chatbot, temperature, max_new_tokens, downvote_btn], None, preprocess=False)
    flag_btn.click(lambda *args: callback.flag(args, flag_option="flag"),
                   [gallery, chatbot, temperature, max_new_tokens, flag_btn], None, preprocess=False)

demo.launch(share=True, enable_queue=True)