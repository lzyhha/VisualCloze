from abc import ABC, abstractmethod
import copy
import json
import logging
import os
from pathlib import Path
import random
from time import sleep
import traceback
import warnings

import h5py
import torch.distributed as dist
from torch.utils.data import Dataset
import yaml
from data.prefix_instruction import degradation_list
from data.data_utils import check_item_graph200k

logger = logging.getLogger(__name__)


class DataBriefReportException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"{self.__class__}: {self.message}"


class ItemProcessor(ABC):
    @abstractmethod
    def process_item(self, data_item, training_mode=False):
        raise NotImplementedError


class MyDataset(Dataset):
    def __init__(self, config_path, item_processor: ItemProcessor, cache_on_disk=False, task_dicts=None):
        logger.info(f"read dataset config from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("DATASET CONFIG:")
        logger.info(self.config)
        
        self.task_dicts = task_dicts

        self.cache_on_disk = cache_on_disk
        if self.cache_on_disk:
            cache_dir = self._get_cache_dir(config_path)
            if dist.get_rank() == 0:
                self._collect_annotations_and_save_to_cache(cache_dir)
            dist.barrier()
            ann, group_indice_range = self._load_annotations_from_cache(cache_dir)
        else:
            cache_dir = None
            ann, group_indice_range = self._collect_annotations()
        
        self.ann = ann
        self.group_indices = {key: list(range(val[0], val[1])) for key, val in group_indice_range.items()}
        self.group_weights = {
            'image_grid_graph200k': 1.0, 
        }
        self.item_processor = item_processor
        
        self.check_item = {
            'image_grid_graph200k': self.check_item_graph200k, 
        }

        logger.info(f"total length: {len(self)}")

    def __len__(self):
        return len(self.ann)
    
    def _collect_annotations(self):
        group_ann = {}
        for meta in self.config["META"]:
            meta_path, meta_type = meta["path"], meta.get("type", "default")
            meta_ext = os.path.splitext(meta_path)[-1]
            if meta_ext == ".json":
                with open(meta_path) as f:
                    meta_l = json.load(f)
            elif meta_ext == ".jsonl":
                meta_l = []
                with open(meta_path) as f:
                    for i, line in enumerate(f):
                        try:
                            meta_l.append(json.loads(line))
                        except json.decoder.JSONDecodeError as e:
                            logger.error(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}")
                            raise e
            else:
                raise NotImplementedError(
                    f'Unknown meta file extension: "{meta_ext}". '
                    f"Currently, .json, .jsonl are supported. "
                    "If you are using a supported format, please set the file extension so that the proper parsing "
                    "routine can be called."
                )
            logger.info(f"{meta_path}, type{meta_type}: len {len(meta_l)}")
            if "ratio" in meta:
                random.seed(0)
                meta_l = random.sample(meta_l, int(len(meta_l) * meta["ratio"]))
                logger.info(f"sample (ratio = {meta['ratio']}) {len(meta_l)} items")
            if "root" in meta:
                for item in meta_l:
                    for path_key in ["path", "image_url", "image", "input_path", "target_path"]:
                        if path_key in item:
                            item[path_key] = os.path.join(meta["root"], item[path_key])
            if meta_type not in group_ann:
                group_ann[meta_type] = []
            group_ann[meta_type] += meta_l

        ann = sum(list(group_ann.values()), start=[])

        group_indice_range = {}
        start_pos = 0
        for meta_type, meta_l in group_ann.items():
            group_indice_range[meta_type] = [start_pos, start_pos + len(meta_l)]
            start_pos = start_pos + len(meta_l)

        return ann, group_indice_range

    def _collect_annotations_and_save_to_cache(self, cache_dir):
        if (Path(cache_dir) / "data.h5").exists() and (Path(cache_dir) / "ready").exists():
            # off-the-shelf annotation cache exists
            warnings.warn(
                f"Use existing h5 data cache: {Path(cache_dir)}\n"
                f"Note: if the actual data defined by the data config has changed since your last run, "
                f"please delete the cache manually and re-run this experiment, or the data actually used "
                f"will not be updated"
            )
            return

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        ann, group_indice_range = self._collect_annotations()

        # when cache on disk, rank0 saves items to an h5 file
        serialized_ann = [json.dumps(_) for _ in ann]
        logger.info(f"start to build data cache to: {Path(cache_dir)}")
        with h5py.File(Path(cache_dir) / "data.h5", "w") as file:
            dt = h5py.vlen_dtype(str)
            h5_ann = file.create_dataset("ann", (len(serialized_ann),), dtype=dt)
            h5_ann[:] = serialized_ann
            file.create_dataset("group_indice_range", data=json.dumps(group_indice_range))
        with open(Path(cache_dir) / "ready", "w") as f:
            f.write("ready")
        logger.info(f"data cache built")

    @staticmethod
    def _get_cache_dir(config_path):
        config_identifier = config_path
        disallowed_chars = ["/", "\\", ".", "?", "!"]
        for _ in disallowed_chars:
            config_identifier = config_identifier.replace(_, "-")
        cache_dir = f"./accessory_data_cache/{config_identifier}"
        return cache_dir

    @staticmethod
    def _load_annotations_from_cache(cache_dir):
        while not (Path(cache_dir) / "ready").exists():
            # cache has not yet been completed by rank 0
            assert dist.get_rank() != 0
            sleep(1)
        cache_file = h5py.File(Path(cache_dir) / "data.h5", "r")
        annotations = cache_file["ann"]
        group_indice_range = json.loads(cache_file["group_indice_range"].asstr()[()])
        return annotations, group_indice_range

    def get_item_func(self, index, group_name=None):
        data_item = self.ann[index]
        if self.cache_on_disk:
            data_item = json.loads(data_item)
        else:
            data_item = copy.deepcopy(data_item)

        return self.item_processor.process_item(data_item, training_mode=True, group_name=group_name)

    def get_item_batch_func(self, index_list, image_type_list=None, context_num=None, group_name=None):
        data_item = [self.ann[index] for index in index_list]
        if self.cache_on_disk:
            data_item = [json.loads(data_item) for data_item in data_item]
        else:
            data_item = [copy.deepcopy(data_item) for data_item in data_item]

        return self.item_processor.process_item(data_item, training_mode=True, image_type_list=image_type_list, context_num=context_num, group_name=group_name)

    def check_item_graph200k(self, index, image_type_list):
        data_item = self.ann[index]
        if self.cache_on_disk:
            data_item = json.loads(data_item)
        else:
            data_item = copy.deepcopy(data_item)

        return check_item_graph200k(data_item, image_type_list)

    def get_context_index(self, index, tried_indices):
        for group_name, indices_this_group in self.group_indices.items():
                if indices_this_group[0] <= index <= indices_this_group[-1]:
                    available_indices = [i for i in indices_this_group if i not in tried_indices]
                    if available_indices:
                        index = random.choice(available_indices)
                        tried_indices.add(index)
                        break
        return index
    
    def get_group_name(self, index):
        for group_name, indices_this_group in self.group_indices.items():
                if indices_this_group[0] <= index <= indices_this_group[-1]:
                    return group_name

    def sample_group(self):
        weights = list(self.group_weights.values())
        groups = list(self.group_weights.keys())
        
        sample = random.choices(groups, weights=weights, k=1)[0]
        
        return sample

    def __getitem__(self, index):
        
        group_name = self.sample_group()
        index = random.choice(self.group_indices[group_name])

        tried_indices = set([index])
        context_num_prob = [0.3, 0.4, 0.3]
        context_num = random.choices([1, 2, 3], weights=context_num_prob)[0]
        task_weight_prob = [task["sample_weight"] for task in self.task_dicts[group_name]]
        block_list = []
        while True:
            task_type = random.choices(self.task_dicts[group_name], weights=task_weight_prob)[0]
            image_type_list = random.choice(task_type["image_list"])
            if not any(block_type in image_type_list for block_type in block_list):
                break
        
        check_item = self.check_item[group_name]

        while True:
            try:
                results = self.get_items(index, check_item, context_num, image_type_list, tried_indices, group_name)
                break
            except Exception as e:
                print(e.with_traceback())
        return results
    
    def get_items(self, index, check_item, context_num, image_type_list, tried_indices, group_name):
        index_list = []
        while len(index_list) < context_num:
            index = self.get_context_index(index, tried_indices)
            if check_item(index, image_type_list):
                index_list.append(index)
        return self.get_item_batch_func(index_list, image_type_list, context_num, group_name)
