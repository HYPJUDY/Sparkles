#!/bin/bash
#https://github.com/DCDmllm/Cheetah/blob/main/DEMON%20Benchmark/scripts/i4_scripts.sh
multi_modal_dialogue="MMCoQA ALFRED"
visual_story_telling_list="AESOP DiDeMoSV FlintstonesSV PororoSV VIST"
visual_relation_inference="IEdit Spot-the-Diff Birds-to-Words CLEVR-Change"
multi_modal_cloze="COMICS_Dialogue COMICS_Panel RecipeQA_VisualCloze RecipeQA_TextCloze"
knowledge_grounded_qa="WebQA TQA MultiModalQA"
text_rich_images_qa="SlideVQA OCR-VQA DocVQA"
multi_image_reasoning="Fashion200K NLVR2 nuscenes VizWiz MIT-States_StateCoherence MIT-States_PropertyCoherence VISION RecipeQA_ImageCoherence"
all_dataset="MMCoQA ALFRED\
 AESOP DiDeMoSV FlintstonesSV PororoSV VIST\
 IEdit Spot-the-Diff Birds-to-Words CLEVR-Change\
 COMICS_Dialogue COMICS_Panel RecipeQA_VisualCloze RecipeQA_TextCloze\
 WebQA TQA MultiModalQA\
 SlideVQA OCR-VQA DocVQA\
 Fashion200K NLVR2 nuscenes VizWiz MIT-States_StateCoherence MIT-States_PropertyCoherence VISION RecipeQA_ImageCoherence"

cfg=/mnt/localdata/Users/yupanhuang/code/Sparkles/eval_configs/sparkles_eval.yaml
i4_dir=/mnt/localdata/Users/yupanhuang/data/I4-Core
result_dir=/mnt/localdata/Users/yupanhuang/models/Sparkles/eval_I4/pretrained_sparkleschat_7b

#for dataset in $all_dataset
for dataset in $multi_modal_dialogue
do
    if [ $dataset = "MMCoQA" ]
    then
        batch=1
    else
        batch=8
    fi
    python /mnt/localdata/Users/yupanhuang/code/Sparkles/evaluate_I4.py --cfg-path $cfg --gpu-id 0 --batch-image $batch --i4-dir $i4_dir --dataset $dataset --result-dir $result_dir
    python calculate_I4.py --i4-dir $i4_dir --dataset $dataset --result-dir $result_dir
done
python i4_score.py --result_dir $result_dir