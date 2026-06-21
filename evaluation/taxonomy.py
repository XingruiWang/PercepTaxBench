import os
import re
import tempfile
from functools import partial
from ..smp import *
import pandas as pd
import random

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE, TaxonomyBench_utils
from ..smp import *
from ..utils import track_progress_rich


class TaxonomyBench(ImageBaseDataset):
    TYPE = "VQA"
    # When ROBUST is True, if the models does not follow the format, all of the response will be treated as answers.
    ROBUST = True

    # DATASET_URL = {
    #     "TaxonomyBench": "./Data/taxonomy.tsv",
    #     "TaxonomyBenchSim": "./Data/taxonomy_sim.tsv",
    #     "Taxonomy_manual": "./Data/taxonomy_manual.tsv",
    # }
    DATASET_URL = {
        "TaxonomyBench": "./Data/taxonomy_real_open_answer.tsv",
        "TaxonomyBenchSim": "./Data/taxonomy_sim_open_answer.tsv",
        # "Taxonomy_manual": "./Data/taxonomy_manual.tsv",
    }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dataset_name = kwargs.get('dataset', 'TaxonomyBench')
        if dataset_name not in self.DATASET_URL:
            dataset_name = 'TaxonomyBench'
        
        self.dataset_utils = TaxonomyBench_utils()
        full_data = self.prepare_tsv(self.DATASET_URL[dataset_name])
        
        # random.seed(42)
        # indices = list(range(len(full_data)))
        # random.shuffle(indices)
        # selected_indices = sorted(indices[:10])
        # self.data = full_data.iloc[selected_indices].reset_index(drop=True)
        self.data = full_data
                
    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)
        
        # Check if url is a local file path
        if osp.exists(url) and osp.isfile(url):
            # It's a local file, just copy it
            if not osp.exists(data_path) or (file_md5 is not None and md5(data_path) != file_md5):
                import shutil
                shutil.copy2(url, data_path)
                update_flag = True
        elif osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
            pass
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(url, data_path)
            update_flag = True

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)
    
    def evaluate(self, eval_file, **judge_kwargs):

        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        all_results = {
            "correct": 0,
            "total": 0,
            "answers": [],
            "format_error": 0,
        }

        for i in tqdm(range(len(lines))):

            line = lines[i]
            index = int(line["index"])

            answers = str(line["answer"])

            try:
                # exact match
                correct = answers.lower() == line["prediction"].lower().strip()
                # if answers == line["prediction"].strip():
                #     correct = True
                # else:
                #     # llm judge
                #     print("LLM judge")
                #     correct = "True" in self.dataset_utils.is_correct(answers, line["prediction"])
                       
                print(f"Index: {index}, Answers: {answers}, Prediction: {line['prediction']}, Correct: {correct}")
            except Exception as e:
                print(e)
                continue

            all_results["answers"].append(
                {
                    "index": index,
                    "correct": correct,
                    "answers": answers,
                    "predict": line["prediction"],
                    "reasoning": "",
                }
            )

            all_results["total"] += 1
            if correct:
                all_results["correct"] += 1

        all_results["score"] = all_results["correct"] / all_results["total"]


        score_pth = get_intermediate_file_path(eval_file, "_score", "json")

        dump(all_results, score_pth)
        return all_results

    def build_prompt(self, line):
        msgs = super().build_prompt(line)

        # for MCQ format
        # instruction = "You are an expert in question answering and obejct recognition. Answer the question with 1-5 words only. Most quesitions are about objects in colored boxes. If the question specifies objects to choose from, answer with ONLY color box name (e.g., 'Red box'). If asked about spatial position, answer with ONLY one word (e.g., 'above', 'below'). Do not explain or elaborate with long sentences."
        # msgs.insert(0, {"type": "text", "value": instruction})


        # for open-ended format
        instruction = (
                    "You are performing strict visual grounding evaluation. "
                    "Return exactly ONE answer (maximum 2 words). "
                    "The bounding box defines the target object for this question. "
                    "If the question asks for an object, "
                    "Name the single object that the bounding box is intended to identify. "
                    "The box specifies the target even if other objects overlap, occlude, or fall inside it. "
                    "Ignore overlapping or foreground objects. "
                    "If the question asks about spatial relations: "
                    "Interpret all relations relative to the object designated by the bounding box. "
                    "Ignore other objects that overlap or partially cover the boxed object. "
                    "Return exactly one direction word: left, right, above, below, in front of, or behind. "
                    "Do not explain. Do not add extra words. Do not include punctuation. Output only the answer."
                )
        msgs.insert(0, {"type": "text", "value": instruction})
        return msgs
