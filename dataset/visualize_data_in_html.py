import os
import json
import tqdm
import argparse
from dataset.config_path import get_Sparkles_path
import html

def visualize_VLtasks_in_html(results, relative_dir='../images', output_file='results.html'):
    page = '<html><body>'

    for result in results:
        evaluation_id = result['evaluation_id']
        page += '<h1>evaluation_id: {}</h1>'.format(evaluation_id)
        page += '<div>'
        if result['label'].lower() == result['predict'].lower():
            page += '<h2 style="color:green">{}</h2>'.format(result['sentence'])
            page += '<p style="color:green">Correct</p>'
        else:
            page += '<h2 style="color:red">{}</h2>'.format(result['sentence'])
            page += '<p style="color:red">Wrong</p>'
        page += '<p>label: {}</p>'.format(result['label'])
        page += '<p>predict: {}</p>'.format(result['predict'])
        page += '<p>prompt: {}</p>'.format(result['prompt'])
        page += '<p>response: {}</p>'.format(result['response'])
        page += '<img src="{}" style="width:200px">'.format(
            os.path.join(relative_dir, result['img_paths'][0].split('/')[-1]))
        page += '<img src="{}" style="width:200px">'.format(
            os.path.join(relative_dir, result['img_paths'][1].split('/')[-1]))
        page += '</div><hr>'

    page += '</body></html>'

    with open(output_file, 'w') as f:
        f.write(page)


def visualize_dialogues_in_html(dialogues, prediction_model='', relative_dir='../images',
                                output_file='dialogue.html', judge_models=None):
    page = '''<!DOCTYPE html>
            <html>
            <head>
            <style>
            body {
                font-family: Arial, sans-serif;
            }

            .dialogue {
                margin-bottom: 1em;
                border: 1px solid #ccc;  
                border-radius: 5px;  
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  
                background-color: #fff;  
                padding: 15px;  
            }

            .utterance {
                margin: 1em;
                clear: both;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                transition: 0.3s;
            }

            .role {
                font-weight: bold;
            }

            .role.user {
                float: right;
                margin: 1em;
            }

            .content.user {
                float: right;
                # background-color: lightyellow;
                # color: green;
                color: #0078ff;
                font-weight: bold;
            }

            .role.assistant {
                float: left;
            }

            .content {
                max-width: 100%;
                padding: 1em;
            }

            .content img {
                height: 50px;
            }

            .user .content, .user .role {
                # float: right;
                margin-left: 40%;
            }

            .assistant .content, .assistant .role {
                float: left;
                margin-right: 40%;
            }
            
             .evaluation {
                margin-bottom: 1em;
                border: 1px solid #ccc;  
                border-radius: 5px;  
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  
                background-color: #fff;  
                padding: 15px;  
                
                display: inline-block;  
                vertical-align: top;  
                # width: 49%;
                width: 100%;  
                margin-right: 1%;  
                box-sizing: border-box;  
            }
            
            .judge_model, .prediction_model {
                font-weight: bold;
            }
            
            .evaluation .scores {
                margin: 1em;
                color: green;
                font-weight: bold;
            }
            
            .evaluation .response {
                margin: 1em;
                clear: both;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                transition: 0.3s;
                padding: 1em;
            }
            
            .images {
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                margin-bottom: 1em;
            }

            .image {
                text-align: center;
                width: 20%;
            }

            .image img {
                height: 200px;
            }

            .image_id, .caption {
                font-size: 0.8em;
            }
            
            .caption {  
              margin-top: 5px;  
              overflow: hidden;  
              text-overflow: ellipsis;  
            }  
              
            button.arrow {  
              right: 5px;  
              bottom: 5px;  
              background: transparent;  
              border: none;  
              font-size: 1.2em;  
              cursor: pointer;  
            }  

            .image-tag {
                color: blue;
                text-decoration: underline;
                position: relative;
            }

            .image-tag img {
                display: none;
                position: absolute;
                top: 1em;
                left: 0;
                height: 200px;
            }

            .image-tag:hover img {
                display: block;
            }

            /* The Modal (background) */
            .modal {
                display: none; /* Hidden by default */
                position: fixed; /* Stay in place */
                z-index: 1; /* Sit on top */
                padding-top: 100px; /* Location of the box */
                left: 0;
                top: 0;
                width: 100%; /* Full width */
                height: 100%; /* Full height */
                overflow: auto; /* Enable scroll if needed */
                background-color: rgb(0,0,0); /* Fallback color */
                background-color: rgba(0,0,0,0.9); /* Black w/ opacity */
            }

            /* Modal Content (Image) */
            .modal-content {
                margin: auto;
                display: block;
                width: 80%;
                max-width: 700px;
            }

            /* Add Animation - Zoom in the Modal */
            .modal-content, #caption { 
                animation-name: zoom;
                animation-duration: 0.6s;
            }

            @keyframes zoom {
                from {transform:scale(0)} 
                to {transform:scale(1)}
            }

            /* The Close Button */
            .close {
                position: absolute;
                top: 15px;
                right: 35px;
                color: #f1f1f1;
                font-size: 40px;
                font-weight: bold;
                transition: 0.3s;
            }

            .close:hover,
            .close:focus {
                color: #bbb;
                text-decoration: none;
                cursor: pointer;
            }

            </style>
            </head>
            <body>
            '''

    for i, item in tqdm.tqdm(enumerate(dialogues)):
        # If this is not the first dialogue, add a line before it
        if i > 0:
            page += '<hr>'

        if 'evaluation_id' in item:
            evaluation_id = item['evaluation_id']
            page += f'<h2>evaluation_id: {evaluation_id}</h2>'
        else:
            evaluation_id = i
        if 'dialogue' in item:
            dialogue = item['dialogue']
        else:
            dialogue = item

        imageid2path = {}
        # Image Section
        page += '<div class="images">'
        for utterance in dialogue:
            if 'images' in utterance:
                for image in utterance['images']:
                    image_id = image["image_id"]
                    image_path = f'<img id="myImg" src="{relative_dir}/{image["image_id"]}.jpg" alt="" onclick="onClick(this)">'
                    imageid2path[image_id] = image_path

                    page += '<div class="image">'
                    page += f'<div class="image_id">Image ID: {image_id}</div>'
                    page += image_path
                    # page += '<div class="caption">' + image["caption"] + '</div>'
                    caption = html.escape(image["caption"])  # escape special HTML characters
                    page += f'<div class="caption" id="caption_{image_id}" data-full-caption="{caption}">' + \
                            caption[:100] + '...</div>'
                    page += f'<button class="arrow" id="arrow_{image_id}" onclick="toggleCaption(\'{image_id}\')">&#8595;</button>'
                    page += '</div>'  # close image div
        page += '</div>'  # close images div

        # Dialogue Section
        page += '<div class="dialogue">'
        page += '<div class="prediction_model">' + "ASSISTANT MODEL: " + prediction_model + '</div>'
        for utterance in dialogue:
            page += '<div class="utterance">'
            # page += '<div class="role ' + utterance['role'] + '">' + utterance['role'] + '</div>'
            if 'content' in utterance:
                content = utterance['role'] + ": " + utterance['content']
                if 'images' in utterance:
                    for image in utterance['images']:
                        image_tag = "<Img><ImageHere></Img>"
                        image_path = imageid2path[image["image_id"]]
                        content = content.replace(image_tag, image_path, 1)

                for image_id, image_path in imageid2path.items():
                    # image_tag = f"IMAGE#{image_id}"
                    image_tag = f"#{image_id}"
                    image_link = f'<span class="image-tag">{image_tag}{image_path}</span>'
                    content = content.replace(image_tag, image_link)

                page += '<div class="content ' + utterance['role'] + '">' + content + '</div>'
            page += '</div>'  # close utterance div
        page += '</div>'  # close dialogue div

        # Evaluation Section
        if judge_models is not None:
            for judge_model in judge_models:
                if judge_model in item:
                    page += '<div class="evaluation">'
                    page += '<div class="judge_model">' + "JUDGE MODEL: " + judge_model + '</div>'
                    page += '<div class="scores">'
                    A1, C1, C2, C3 = item[judge_model]['A1_C123_scores']
                    page += f'<div class="A1_C123_scores">Turn 1 A1: {A1}, C1: {C1}, C2: {C2}, C3: {C3}</div>'
                    A2, C1, C2, C3 = item[judge_model]['A2_C123_scores']
                    page += f'<div class="A2_C123_scores">Turn 2 A2: {A2}, C1: {C1}, C2: {C2}, C3: {C3}</div>'
                    page += '</div>'
                    # page += '<div class="response">' + item[judge_model]['response'] + '</div>'
                    # need to replace the double quote character (`"`) with `&quot;` for the page to render correctly
                    response = html.escape(item[judge_model]["response"])  # escape special HTML characters
                    page += f'<div class="response" id="response_{evaluation_id}_{judge_model}" data-full-response="{response}">' + response[:100] + '...</div>'
                    page += f'<button class="arrow" id="arrow_{evaluation_id}_{judge_model}" onclick="toggleResponse(\'{evaluation_id}\', \'{judge_model}\')">&#8595;</button>'
                    page += '</div>'

    page += '''
        <div id="myModal" class="modal">
          <span class="close">&times;</span>
          <img class="modal-content" id="img01">
        </div>

        <script>
        function onClick(element) {
          document.getElementById("img01").src = element.src;
          document.getElementById("myModal").style.display = "block";
        }

        var span = document.getElementsByClassName("close")[0];
        span.onclick = function() { 
          document.getElementById("myModal").style.display = "none";
        }
        
        function toggleCaption(image_id) {  
            var caption = document.getElementById('caption_' + image_id);  
            var arrow = document.getElementById('arrow_' + image_id);  
          
            var full_caption = caption.getAttribute('data-full-caption');  
            var short_caption = full_caption.slice(0, 100) + '...';  
          
            if (caption.innerHTML === short_caption) {  
                caption.innerHTML = full_caption;  
                arrow.innerHTML = '&#8593;';  
            } else {  
                caption.innerHTML = short_caption;  
                arrow.innerHTML = '&#8595;';  
            }  
        }
        
        function toggleResponse(evaluation_id, judge_model) {  
            var response = document.getElementById('response_' + evaluation_id + '_' + judge_model);  
            var arrow = document.getElementById('arrow_' + evaluation_id + '_' + judge_model);  
          
            var full_response = response.getAttribute('data-full-response');  
            var short_response = full_response.slice(0, 100) + '...';  
          
            if (response.innerHTML === short_response) {  
                response.innerHTML = full_response;  
                arrow.innerHTML = '&#8593;';  
            } else {  
                response.innerHTML = short_response;  
                arrow.innerHTML = '&#8595;';  
            }  
        }  

        </script>

        </body>
        </html>
        '''

    with open(output_file, 'w') as f:
        f.write(page)
    print(f'Wrote to {output_file}')


def read_and_visualize(path, visualize_dialogues=True, model_name="gpt4", judge_models=None):
    html_path = path.replace(".json", ".html")
    with open(path, 'r') as fr:
        data = json.load(fr)
    if visualize_dialogues:
        visualize_dialogues_in_html(data, prediction_model=model_name, output_file=html_path, judge_models=judge_models)
    else:
        visualize_VLtasks_in_html(data, output_file=html_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sparkles_root", type=str, default=None, help=(""))
    parser.add_argument("--visualize_eval_results", action="store_true", help="")
    parser.add_argument("--visualize_SparklesDialogue", action="store_true", help="")
    args = parser.parse_args()

    sparkles_root, SparklesDialogueCC_root, SparklesDialogueVG_root, SparklesDialogueCC_path, \
    SparklesDialogueVG_path, SparklesEval_path, BISON_path, NLVR2_path, statistics_dir = \
        get_Sparkles_path(args.sparkles_root)

    if args.visualize_SparklesDialogue:
        read_and_visualize(SparklesDialogueVG_path)
        read_and_visualize(SparklesDialogueCC_path)

    if args.visualize_eval_results:
        # The following JSON files are generated by evaluate.py, their corresponding HTML files have
        # also been generated within evaluate.py by calling functions from this file.
        # The following code illustrates how to solely generate HTML files from the JSON files.
        results_path = os.path.join(os.path.dirname(SparklesEval_path), "..", "results")
        read_and_visualize(os.path.join(results_path, "SparklesEval_models_pretrained_sparkleschat_7b.json"),
                           model_name="sparkleschat_7b", judge_models=["gpt-4"])
        read_and_visualize(os.path.join(results_path, "SparklesEval_models_pretrained_minigpt4_7b.json"),
                           model_name="minigpt4_7b", judge_models=["gpt-4"])

        results_path = os.path.join(os.path.dirname(BISON_path), "..", "results")
        read_and_visualize(os.path.join(results_path, "BISON_models_pretrained_sparkleschat_7b_acc0.567.json"),
                           visualize_dialogues=False)
        read_and_visualize(os.path.join(results_path, "BISON_models_pretrained_minigpt4_7b_acc0.460.json"),
                           visualize_dialogues=False)

        results_path = os.path.join(os.path.dirname(NLVR2_path), "..", "results")
        read_and_visualize(os.path.join(results_path, "NLVR2_models_pretrained_minigpt4_7b_acc0.513.json"),
                           visualize_dialogues=False)
        read_and_visualize(os.path.join(results_path, "NLVR2_models_pretrained_sparkleschat_7b_acc0.580.json"),
                           visualize_dialogues=False)

    print('Done!')