# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score.parser import extract_answer
from verl.utils.reward_score.grader import math_equal
import torch
from collections import defaultdict
import re
from multiprocessing import Process, Queue

def _validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    validation_passed = True

    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'boxed_start': (r'\\boxed\{.*?\}', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        if tag_name == 'boxed_start':
            match = re.findall(tag_str, processed_str)
            count = len(match)
            pos = re.search(tag_str, processed_str)
            if pos is not None:
                positions[tag_name] = re.search(tag_str, processed_str).start()
            else:
                positions[tag_name] = -1
        else:
            count = processed_str.count(tag_str)
            positions[tag_name] = processed_str.find(tag_str)

        if count != expected_count:
            validation_passed = False

    misplace_think = positions.get('think_start') > positions.get('think_end') or positions.get('think_end') > positions.get('answer_start')
    misplace_answer = positions.get('answer_start') > positions.get('boxed_start') or positions.get('boxed_start') > positions.get('answer_end')
    missing_format = not processed_str.startswith('<think>') or not processed_str.endswith('</answer>')
    if (misplace_think
            or misplace_answer or missing_format):
        validation_passed = False
    else:
        pass

    return validation_passed

def _math_worker(q, prediction, reference):
    result = math_equal(prediction=prediction, reference=reference, timeout=False)
    q.put(result)


def _extract_worker(q, model_output):
    result = extract_answer(pred_str=model_output, data_name="math")
    q.put(result)

def _format_worker(q, model_output):
    result = _validate_response_structure(processed_str=model_output)
    q.put(result)

def math_equal_subprocess(prediction, reference, timeout_seconds=10):
    q = Queue()
    p = Process(target=_math_worker, args=(q, prediction, reference))
    p.start()

    p.join(timeout=timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join()
        return False

    try:
        return q.get_nowait()
    except Exception as e:
        return False
    
def extract_answer_subprocess(model_output, timeout_seconds=10):
    q = Queue()
    p = Process(target=_extract_worker, args=(q, model_output))
    p.start()

    p.join(timeout=timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join()
        return ""

    try:
        return q.get_nowait()
    except Exception as e:
        return ""
    
def format_subprocess(model_output, timeout_seconds=10):
    q = Queue()
    p = Process(target=_format_worker, args=(q, model_output))
    p.start()

    p.join(timeout=timeout_seconds)

    if p.is_alive():
        p.terminate()
        p.join()
        return False

    try:
        return q.get_nowait()
    except Exception as e:
        return False

def base_model_accuracy_reward(queue, sequences, answers, *args, **kwargs):
    scores = []
    for sequence, answer in zip(sequences, answers):
        format_correct = format_subprocess(model_output=sequence)

        ext_answer = extract_answer_subprocess(model_output=sequence)
        box_match = 0.0
        if math_equal_subprocess(prediction=ext_answer, reference=answer) and format_correct:
            box_match = 1.0

        scores.append(box_match)

    if queue is not None:
        queue.put(scores)

    return scores



def format_reward(sequences,*args, **kwargs):
    """
    Reward function that checks if the completion has a specific format.

    Args:
        queue: parallel queue
        sequences: A list of sequences, where each completion is a tuple containing a list of dictionaries.
                     Each dictionary should have a "content" key with the text to be checked.

    Returns:
        A list of floats, where each float is 1.0 if the corresponding completion matches the required format,
        and 0.0 otherwise.

    Raises:
        ValueError: If the input sequences are not in the expected format.
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    if not isinstance(sequences, list):
        raise ValueError("Input sequences must be a list.")

    scores = []
    for completion in sequences:
        if re.match(pattern, completion, re.DOTALL | re.MULTILINE):
            scores.append(1.0)
        else:
            scores.append(0.0)


    return scores


class MindspeedRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.rule_verifier_function = {
            "format": format_reward,
            "base_acc": base_model_accuracy_reward}
        self.verifier_function=["base_acc"]

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        completions = []
        valid_response_length_list=[]
        ground_truths=[]

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['ground_truth']


            completions.append(response_str)
            valid_response_length_list.append(valid_response_length)
            ground_truths.append(ground_truth)

        ##################### 改造 ###########################
        rewards = [0.0] * len(ground_truths)
        for idx, fun_verifier in enumerate(self.verifier_function):
            scores = self.rule_verifier_function[fun_verifier](queue=None, sequences=completions, answers=ground_truths)
            rewards = [all_score + tmp_score for all_score, tmp_score in zip(rewards, scores)]
        print(f"grpo/base_acc_rewards/mean :{rewards}")
        for i in range(len(data)):
            reward_tensor[i, valid_response_length_list[i] - 1] = rewards[i]
        # from switch_print import print_feature
        # print_feature(scores, "rewards")
        ##################### 改造 ###########################
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
