# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import argparse
import itertools
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Optional

import torch
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from .textvqa_eval import TextVQAAccuracyEvaluator
from tqdm import tqdm

ds_collections = {
    'vqav2_val': {
        'train': 'eval/vlm/data/vqav2/vqav2_train.jsonl',
        'test': 'eval/vlm/data/vqav2/vqav2_val.jsonl',
        'question': 'eval/vlm/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'eval/vlm/data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'eval/vlm/data/vqav2/vqav2_train.jsonl',
        'test': 'eval/vlm/data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'eval/vlm/data/okvqa/okvqa_train.jsonl',
        'test': 'eval/vlm/data/okvqa/okvqa_val.jsonl',
        'question': 'eval/vlm/data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'eval/vlm/data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'eval/vlm/data/textvqa/textvqa_train.jsonl',
        'test': 'eval/vlm/data/textvqa/textvqa_val.jsonl',
        'question': 'eval/vlm/data/textvqa/textvqa_val_questions.json',
        'annotation': 'eval/vlm/data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val_ocr': {
        'train': 'eval/vlm/data/textvqa/textvqa_train.jsonl',
        'test': 'eval/vlm/data/textvqa/textvqa_val_llava.jsonl',
        'question': 'eval/vlm/data/textvqa/textvqa_val_questions.json',
        'annotation': 'eval/vlm/data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'eval/vlm/data/vizwiz/vizwiz_train.jsonl',
        'test': 'eval/vlm/data/vizwiz/vizwiz_val.jsonl',
        'question': 'eval/vlm/data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'eval/vlm/data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'eval/vlm/data/vizwiz/vizwiz_train.jsonl',
        'test': 'eval/vlm/data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'train': 'eval/vlm/data/docvqa/train.jsonl',
        'test': 'eval/vlm/data/docvqa/val.jsonl',
        'annotation': 'eval/vlm/data/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'eval/vlm/data/docvqa/train.jsonl',
        'test': 'eval/vlm/data/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'train': 'eval/vlm/data/chartqa/train_human.jsonl',
        'test': 'eval/vlm/data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'eval/vlm/data/chartqa/train_augmented.jsonl',
        'test': 'eval/vlm/data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'eval/vlm/data/gqa/train.jsonl',
        'test': 'eval/vlm/data/gqa/test_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'gqa_testdev_llava': {
        'train': 'eval/vlm/data/gqa/train.jsonl',
        'test': 'eval/vlm/data/gqa/llava_gqa_testdev_balanced_qwen_format.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'eval/vlm/data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'eval/vlm/data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'eval/vlm/data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'eval/vlm/data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'eval/vlm/data/ai2diagram/train.jsonl',
        'test': 'eval/vlm/data/ai2diagram/test_vlmevalkit.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'infographicsvqa_val': {
        'train': 'eval/vlm/data/infographicsvqa/train.jsonl',
        'test': 'eval/vlm/data/infographicsvqa/val.jsonl',
        'annotation': 'eval/vlm/data/infographicsvqa/infographicsVQA_val_v1.0_withQT.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'infographicsvqa_test': {
        'train': 'eval/vlm/data/infographicsvqa/train.jsonl',
        'test': 'eval/vlm/data/infographicsvqa/test.jsonl',
        'annotation': 'eval/vlm/data/infographicsvqa/infographicsVQA_test_v1.0.json',
        'metric': None,
        'max_new_tokens': 100,
    }
}

DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
DEFAULT_DATA_ROOT = os.path.join(DEFAULT_PROJECT_ROOT, "eval", "vlm", "data")
DEFAULT_IMAGE_ROOT = os.path.join(DEFAULT_PROJECT_ROOT, "eval", "vlm")
resolved_ds_collections = ds_collections


def _resolve_existing_path(path, *, data_root, project_root):
    if path is None:
        return None

    if os.path.isabs(path):
        return path

    candidates = [path]
    if project_root:
        candidates.append(os.path.join(project_root, path))
    if data_root and path.startswith("eval/vlm/data/"):
        relative_path = path[len("eval/vlm/data/") :]
        candidates.append(os.path.join(data_root, relative_path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def _resolve_image_path(image_path, *, image_root, project_root, dataset_file):
    if os.path.isabs(image_path):
        return image_path

    candidates = [image_path]
    if image_root:
        candidates.append(os.path.join(image_root, image_path))
    if project_root:
        candidates.append(os.path.join(project_root, "eval", "vlm", image_path))
    if dataset_file:
        candidates.append(os.path.join(os.path.dirname(dataset_file), image_path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _resolve_dataset_paths(collections, *, data_root, project_root):
    resolved = {}
    for name, config in collections.items():
        config_resolved = dict(config)
        for key in ("train", "test", "question", "annotation"):
            if key in config_resolved:
                config_resolved[key] = _resolve_existing_path(
                    config_resolved[key],
                    data_root=data_root,
                    project_root=project_root,
                )
        resolved[name] = config_resolved
    return resolved


def _validate_dataset_paths(dataset_names, collections, *, few_shot):
    missing = []
    for dataset_name in dataset_names:
        config = collections[dataset_name]
        required_keys = ["test"]
        if few_shot > 0:
            required_keys.append("train")
        if config.get("metric") == "vqa_score":
            required_keys.extend(["question", "annotation"])
        elif config.get("metric") == "anls":
            required_keys.append("annotation")

        for key in required_keys:
            path = config.get(key)
            if path and not os.path.exists(path):
                missing.append((dataset_name, key, path))

    if missing:
        details = "\n".join(
            f"  - dataset={dataset_name}, key={key}, path={path}"
            for dataset_name, key, path in missing
        )
        raise FileNotFoundError(
            "Missing evaluation data files.\n"
            f"{details}\n"
            "Set --data-root/--project-root (or BAGEL_EVAL_DATA_ROOT/BAGEL_PROJECT_ROOT) "
            "to the directory that contains eval/vlm/data."
        )


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    images = [_['images'] for _ in batches]
    conversations = [_['conversations'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    return questions, images, conversations, question_ids, annotations


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, image_root=None, project_root=None):
        self.test_path = test
        self.test = open(test, encoding="utf-8").readlines()
        self.prompt = prompt
        self.few_shot = few_shot
        self.image_root = image_root
        self.project_root = project_root
        if few_shot > 0:
            self.train = open(train, encoding="utf-8").readlines()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)

        few_shot_prompt = ''
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += self.prompt.format(
                    sample['image'],
                    sample['question']) + f" {sample['answer']}"
        
        image = Image.open(
            _resolve_image_path(
                image,
                image_root=self.image_root,
                project_root=self.project_root,
                dataset_file=self.test_path,
            )
        ).convert('RGB')
        images = [image]
        
        if len(self.prompt) != 0:
            question = question + ' ' + self.prompt

        images, conversation = process_conversation(images, question)

        return {
            'question_id': question_id,
            'question': question,
            'images': images,
            'conversations': conversation,
            'annotation': annotation
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(response):
    response = response.strip().split('.')[0].split(
        ',')[0].split('!')[0].lower()
    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]
    response = response.strip()
    return response


def evaluate_chat_model():
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    ai2d_prompt = ''
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'ai2d' in ds_name:
            input_prompt = ai2d_prompt
        elif 'infographicsvqa' in ds_name:
            input_prompt = infovqa_prompt
        else:
            input_prompt = base_prompt

        dataset = VQADataset(
            train=resolved_ds_collections[ds_name]['train'],
            test=resolved_ds_collections[ds_name]['test'],
            prompt=input_prompt,
            few_shot=args.few_shot,
            image_root=args.image_root,
            project_root=args.project_root,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        outputs = []
        for _, (questions, images, conversations, question_ids, annotations) in tqdm(enumerate(dataloader)):
            pred = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images[0], # batch=1
                prompt=conversations[0], # batch=1
                max_length=ds_collections[ds_name]['max_new_tokens'], # TODO: how to use ds_collections[ds_name]['min_new_tokens']
            )
            answers = [pred]

            for question, question_id, answer, annotation in zip(questions, question_ids, answers, annotations):
                if ds_name in ['vqav2_val', 'vqav2_testdev', 'okvqa_val', 'textvqa_val',
                               'vizwiz_val', 'textvqa_val_ocr']:
                    outputs.append({
                        'question': question,
                        'question_id': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['docvqa_val', 'infographicsvqa_val', 'gqa_testdev', 'ocrvqa_val',
                                 'ocrvqa_test', 'gqa_testdev_llava', 'infographicsvqa_test',]:
                    outputs.append({
                        'question': question,
                        'questionId': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['ai2diagram_test']:
                    outputs.append({
                        'question': question,
                        'image': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['chartqa_test_human', 'chartqa_test_augmented']:
                    outputs.append({
                        'question': question,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['docvqa_test']:
                    outputs.append({
                        'questionId': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['vizwiz_test']:
                    outputs.append({
                        'image': question_id.replace('data/vizwiz/test/', ''),
                        'answer': answer,
                    })
                else:
                    raise NotImplementedError

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            results_file = 'results.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

            if ds_collections[ds_name]['metric'] == 'vqa_score':
                evaluator = TextVQAAccuracyEvaluator()
                annotation = json.load(open(resolved_ds_collections[ds_name]['annotation'], 'r', encoding='utf-8'))['annotations']
                question_id2answers = {}
                for item in annotation:
                    question_id = item['question_id']
                    answers = [answer['answer'] for answer in item['answers']]
                    question_id2answers[question_id] = answers
                for item in merged_outputs:
                    item['pred_answer'] = item['answer']
                    item['gt_answers'] = question_id2answers[item['question_id']]
                accuracy = evaluator.eval_pred_list(merged_outputs)
                print(ds_name, "\nvqa_score")
                print(accuracy)
                summaries.append(accuracy)

            elif ds_collections[ds_name]['metric'] == 'anls':
                json.dump(merged_outputs,
                          open(results_file, 'w'),
                          ensure_ascii=False)
                print('python eval/vqa/infographicsvqa_eval.py -g ' +
                      resolved_ds_collections[ds_name]['annotation'] + ' -s ' +
                      results_file)
                os.system('python eval/vqa/infographicsvqa_eval.py -g ' +
                          resolved_ds_collections[ds_name]['annotation'] + ' -s ' +
                          results_file)
            elif ds_collections[ds_name]['metric'] == 'relaxed_accuracy':
                relaxed_accuracy = evaluate_relaxed_accuracy(merged_outputs)
                print(ds_name, "\relaxed_accuracy")
                print(relaxed_accuracy)
                summaries.append(relaxed_accuracy)
            elif ds_collections[ds_name]['metric'] == 'accuracy':
                if 'gqa' in ds_name:
                    dst_file = 'eval/vlm/data/gqa/testdev_balanced_predictions.json'
                    print('python eval/vlm/eval/vqa/convert_gqa_for_eval.py --src ' +
                          results_file + ' --dst ' + dst_file)
                    python_path = 'python'
                    os.system(python_path + ' eval/vlm/eval/vqa/convert_gqa_for_eval.py --src ' +
                              results_file + ' --dst ' + dst_file)
                    command = f'cd ./eval/vlm/data/gqa/ && {python_path} eval.py --tier testdev_balanced && cd ../../../../'
                    print(command)
                    accuracy = subprocess.check_output(command, shell=True, universal_newlines=True)
                else:
                    accuracy = {'accuracy': evaluate_exact_match_accuracy(merged_outputs)}
                print(ds_name, "\naccuracy")
                print(accuracy)
                summaries.append(accuracy)

        torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        writer = open(os.path.join(args.out_dir, "results.txt"), 'w')
        print(f"write results to file {os.path.join(args.out_dir, 'results.txt')}")
        output_content = ""
        for item in summaries:
            output_content += f"{item}\n"
        writer.write(output_content)
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str,
                        default='okvqa_val,textvqa_val,vizwiz_val,ai2diagram_test,gqa_testdev_llava')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--project-root', type=str, default=os.getenv('BAGEL_PROJECT_ROOT', DEFAULT_PROJECT_ROOT))
    parser.add_argument('--data-root', type=str, default=os.getenv('BAGEL_EVAL_DATA_ROOT', DEFAULT_DATA_ROOT))
    parser.add_argument('--image-root', type=str, default=os.getenv('BAGEL_EVAL_IMAGE_ROOT', DEFAULT_IMAGE_ROOT))
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'
    unknown_datasets = [ds for ds in args.datasets if ds not in ds_collections]
    assert not unknown_datasets, f"Unknown datasets: {unknown_datasets}"

    resolved_ds_collections = _resolve_dataset_paths(
        ds_collections,
        data_root=args.data_root,
        project_root=args.project_root,
    )
    _validate_dataset_paths(args.datasets, resolved_ds_collections, few_shot=args.few_shot)

    dist_initialized = False

    try:
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            rank=int(os.getenv('RANK', '0')),
        )
        dist_initialized = True

        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

        model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
        image_transform = build_transform()

        total_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f'[test] total_params: {total_params}B')

        evaluate_chat_model()
    finally:
        if dist_initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
