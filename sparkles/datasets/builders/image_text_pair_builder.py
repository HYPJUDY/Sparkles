import os
import logging
import warnings

from sparkles.common.registry import registry
from sparkles.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from sparkles.datasets.datasets.laion_dataset import LaionDataset
from sparkles.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from sparkles.datasets.datasets.multimodal_dialogue_dataset import MultimodalDialogueDataset

@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[build_info.ann_paths],
            vis_root=build_info.vis_root,
        )

        return datasets


class MultimodalDialogueBuilder(BaseDatasetBuilder):
    train_dataset_cls = MultimodalDialogueDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/defaults.yaml"}

    def build_datasets(self):
        self.build_processors()

        datasets = dict()
        split = "train"

        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            anno_path=self.config.anno_path,
            root_dir=self.config.root_dir,
            llama_model=self.config.llama_model,
            img_root=self.config.img_root,
        )

        return datasets


registry.register_builder("LLaVA_description")(MultimodalDialogueBuilder)
registry.register_builder("LLaVA_reasoning")(MultimodalDialogueBuilder)

registry.register_builder("SparklesDialogueCC_turn1_1img")(MultimodalDialogueBuilder)
registry.register_builder("SparklesDialogueCC_turn1_2img")(MultimodalDialogueBuilder)
registry.register_builder("SparklesDialogueCC_turn1_3img")(MultimodalDialogueBuilder)
registry.register_builder("SparklesDialogueCC_turn2_2img")(MultimodalDialogueBuilder)
registry.register_builder("SparklesDialogueCC_turn2_3img")(MultimodalDialogueBuilder)
registry.register_builder("SparklesDialogueCC_turn2_4img")(MultimodalDialogueBuilder)

registry.register_builder("SparklesDialogueVG_turn1_2img")(MultimodalDialogueBuilder)
registry.register_builder("SparklesDialogueVG_turn1_3img")(MultimodalDialogueBuilder)
registry.register_builder("SparklesDialogueVG_turn2_3img")(MultimodalDialogueBuilder)
registry.register_builder("SparklesDialogueVG_turn2_4img")(MultimodalDialogueBuilder)