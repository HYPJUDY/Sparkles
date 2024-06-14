import os

def get_Sparkles_path(sparkles_root=None):
    sparkles_root = "/mnt/localdata/Users/yupanhuang/data/Sparkles/" if sparkles_root is None else sparkles_root
    SparklesDialogueCC_root = os.path.join(sparkles_root, "data", "SparklesDialogueCC")
    SparklesDialogueVG_root = os.path.join(sparkles_root, "data", "SparklesDialogueVG")
    SparklesDialogueCC_path = os.path.join(SparklesDialogueCC_root, "annotations", "SparklesDialogueCC.json")
    SparklesDialogueVG_path = os.path.join(SparklesDialogueVG_root, "annotations", "SparklesDialogueVG.json")
    SparklesEval_path = os.path.join(sparkles_root, "evaluation", "SparklesEval", "annotations",
                                     "sparkles_evaluation_sparkleseval_annotations.json")
    BISON_path = os.path.join(sparkles_root, "evaluation", "BISON", "annotations",
                              "sparkles_evaluation_bison_annotations.json")
    NLVR2_path = os.path.join(sparkles_root, "evaluation", "NLVR2", "annotations",
                              "sparkles_evaluation_nlvr2_annotations.json")
    statistics_dir = os.path.join(sparkles_root, "assets", "statistics")
    return sparkles_root, SparklesDialogueCC_root, SparklesDialogueVG_root, SparklesDialogueCC_path, \
           SparklesDialogueVG_path, SparklesEval_path, BISON_path, NLVR2_path, statistics_dir
