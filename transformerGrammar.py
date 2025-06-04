# transformerGrammar.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)
# The question was created by Haoyu Du (duhy@shanghaitech.edu.cn).

import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import numpy as np
import subprocess
try:
    cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

    os.system('echo $CUDA_VISIBLE_DEVICES')
except:
    pass
import util

import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, PreTrainedModel
from transformers.models.gpt_neo import GPTNeoConfig, GPTNeoForCausalLM


class InvalidTreeError(Exception):
    pass


def mapping_function(example: dict) -> dict:
    """
    Question:
        Your task is to return the processed input, processed output, attention mask, and absolute positions of the action sequence for valid actions sequence. The following order may be your implementation order:

            1. Check whether the given action sequence is a valid sequence to generate a legal parse tree. If it is invalid, please raise an InvalidTreeError Exception.
            2. The processed input: a list of strings. It should duplicate all closing nonterminals in the given action sequence.
            3. The processed output: a list of strings. It should insert '<pad>' after all closing nonterminals in the given action sequence.
            4. The absolute positions: a list of integers. The absolute position of each token is defined as the depth of it in the tree.
            5. The attention mask: a 2d torch tensor. This is the attention mask with STACK/COMPOSE attention. The attention mask of '</s>' is all 0s.

        HINT: It is guaranteed that the first item of input is '<s>' (beginning of sequence), and the last item of input is '</s>' (end of sequence). The absolute positions of both '<s>' and '</s>' are 0 in this question.
    
    Args:
        example (dict): The example to process. It has the following fields:
            - actions (List[str]): The action sequence. It is a list of strings which can be regarded as an action sequence for generative transition-based parsing.

    Return:
        mapped (dict): The mapped example. It has the following fields:
            - inputs (List[str]): The processed input. A list of tokens for the input.
            - labels (List[str]): The processed output. A list of tokens for the expected output.
            - position_ids (List[int]): The absolute positions. A list of integers representing the absolute position of each token in the input.
            - attention_mask (torch.Tensor): The attention mask. Shape: (len(input), len(input)). A 2D tensor representing the attention mask for the input sequence. 1 for valid tokens, 0 for padding tokens.

    Example:
        >>> mapping_function({"actions": ["<s>", "(S", "(NP", "the", "blue", "bird", "NP)", "(VP", "sings", "VP)", "S)", "</s>"]})
        {
            'inputs': ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', 'NP)', '(VP', 'sings', 'VP)', 'VP)', 'S)', 'S)', '</s>'],
            'labels': ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', '<pad>', '(VP', 'sings', 'VP)', '<pad>', 'S)', '<pad>', '</s>'],
            'position_ids': [0, 0, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 0, 0, 0],
            'attention_mask': tensor([[...]])
        }
    """

    """YOUR CODE HERE"""
    # util.raiseNotDefined()
    
    action_sequence = example["actions"]
    sequence_length = len(action_sequence)
    
    # 验证：长度以及起始结束符
    if sequence_length < 2 or action_sequence[0] != "<s>" or action_sequence[-1] != "</s>":
        raise InvalidTreeError("error code 1")
    
    # 验证：括号
    validation_stack = []
    for token_index, token in enumerate(action_sequence):
        # 检查并处理开放非终结符 "(X"
        if token[0] == '(' and token[-1] != ')':
            if (name := token[1:]) and not any(c in name for c in '()'):
                validation_stack.append((name, token_index))
            else:
                raise InvalidTreeError(f"Invalid open non-terminal: '{token}' at position {token_index}")
        
        # 检查并处理关闭非终结符 "X)"
        elif token[-1] == ')' and token[0] != '(':
            if not validation_stack:
                raise InvalidTreeError(f"Unmatched close: '{token}' at position {token_index}")
            
            name = token[:-1]
            if name != (expected := validation_stack.pop()[0]):
                raise InvalidTreeError(
                    f"Mismatched parentheses at {token_index}: Expected '{expected})' "
                    f"but found '{token}'"
                )
        
        # 处理终结符（必须不包含括号）
        else:
            if '(' in token or ')' in token:
                raise InvalidTreeError(f"Invalid terminal: '{token}' at position {token_index}")

    # 检查未关闭的非终结符
    if unclosed := [f"'{name}' (opened at {idx})" for name, idx in validation_stack]:
        raise InvalidTreeError(f"Unclosed non-terminals: {', '.join(unclosed)}")
    
    # 构建处理后的输入序列并记录映射关系
    processed_inputs = []
    index_mapping = []

    for orig_idx, token in enumerate(action_sequence):
        # 添加原始标记
        processed_inputs.append(token)
        index_mapping.append((orig_idx, False))  # 标记为非重复项
        
        # 对于关闭非终结符，添加重复标记
        if token.endswith(")") and token not in ("<s>", "</s>"):
            processed_inputs.append(token)
            index_mapping.append((orig_idx, True))  # 标记为重复项

    processed_length = len(processed_inputs)

    # 建立原始索引到处理后索引的映射
    original_to_first_occurrence = {}
    original_to_duplicate = {}

    for proc_idx, (orig_idx, is_duplicate) in enumerate(index_mapping):
        if not is_duplicate:
            # 只记录首次出现的索引
            if orig_idx not in original_to_first_occurrence:
                original_to_first_occurrence[orig_idx] = proc_idx
        else:
            # 记录重复项的索引
            original_to_duplicate[orig_idx] = proc_idx
    
    span_stack = []
    tree_spans = []

    for idx, token in enumerate(action_sequence):
        if token.startswith("(") and not token.endswith(")"):
            span_stack.append((token[1:], idx))
        elif token.endswith(")") and not token.startswith("(") and token not in ("<s>", "</s>"):
            if not span_stack or span_stack[-1][0] != token[:-1]:
                raise InvalidTreeError(f"位置{idx}的括号不匹配: 得到{token}, 期望{span_stack[-1][0] if span_stack else None}")
            tree_spans.append({"start_index": span_stack.pop()[1], "end_index": idx, "children": []})

    if span_stack:
        raise InvalidTreeError("未闭合的跨度")
    if not tree_spans:
        raise InvalidTreeError("未找到非终结符跨度")

    tree_spans.sort(key=lambda s: s["start_index"])
    span_stack = []
    child_positions = set()

    for span in tree_spans:
        while span_stack and not (span["start_index"] > span_stack[-1]["start_index"] and span["end_index"] < span_stack[-1]["end_index"]):
            span_stack.pop()
        if span_stack:
            span_stack[-1]["children"].append(span)
            child_positions.add((span["start_index"], span["end_index"]))
        span_stack.append(span)

    root_candidates = [s for s in tree_spans if (s["start_index"], s["end_index"]) not in child_positions]
    if len(root_candidates) != 1:
        raise InvalidTreeError("必须存在唯一根节点")
    
    def validate_span_has_terminal(span, action_sequence):
        """验证跨度是否包含至少一个直接终结符"""
        for j in range(span["start_index"] + 1, span["end_index"]):
            token = action_sequence[j]
            if not token.startswith("(") and not token.endswith(")"):
                return True
        return False

    def is_token_in_child(j, child_ranges):
        """检查token位置是否在任何子跨度内"""
        return any(start < j < end for start, end in child_ranges)

    # 7 验证每个跨度至少包含一个直接终结符
    for span in tree_spans:
        if not validate_span_has_terminal(span, action_sequence):
            s, e = span["start_index"], span["end_index"]
            raise InvalidTreeError(f"在位置({s},{e})的跨度中没有直接终结符")

    # 构建处理后层面的跨度并建立映射
    processed_spans = []
    span_position_map = {}

    for span in tree_spans:
        orig_start, orig_end = span["start_index"], span["end_index"]
        new_span = {
            "start": original_to_first_occurrence[orig_start],
            "end": original_to_first_occurrence[orig_end],
            "duplicate": original_to_duplicate[orig_end],
            "children": [],
            "orig_start": orig_start,
            "orig_end": orig_end
        }
        processed_spans.append(new_span)
        span_position_map[(orig_start, orig_end)] = new_span

    # 9 填充子节点信息
    for span in tree_spans:
        parent = span_position_map[(span["start_index"], span["end_index"])]
        parent["children"] = [
            span_position_map[(c["start_index"], c["end_index"])]
            for c in span["children"]
        ]

    attention_indices = {}
    for span in processed_spans:
        p_start, p_end = span["start"], span["end"]
        o_start, o_end = span["orig_start"], span["orig_end"]
        
        indices_set = {p_start, p_end}
        # 添加子节点结束位置
        indices_set.update(child["end"] for child in span["children"])
        
        # 收集子跨度范围用于快速检查
        child_ranges = [(c["orig_start"], c["orig_end"]) for c in span["children"]]
        
        # 添加未被子节点覆盖的终结符
        for j in range(o_start + 1, o_end):
            token = action_sequence[j]
            is_terminal = not (token.startswith("(") or token.endswith(")"))
            if is_terminal and not is_token_in_child(j, child_ranges):
                indices_set.add(original_to_first_occurrence[j])
        
        attention_indices[p_end] = indices_set
    
    # 构建注意力掩码矩阵
    attention_mask = torch.zeros((processed_length, processed_length), dtype=torch.float)
    seq_len = processed_length
    attention_mask = torch.zeros((seq_len, seq_len), dtype=torch.float)
    
    active_tokens = {0}  # 初始化为<s>的索引(总是0)
    
    # 构建位置索引集合
    open_positions = {span["start"] for span in processed_spans}    # 开放标记位置
    close_positions = {span["end"] for span in processed_spans}    # 首次关闭标记位置
    duplicate_positions = {span["duplicate"] for span in processed_spans}  # 重复关闭标记位置
    
    # 构建关闭位置到跨度的映射
    close_to_span = {span["end"]: span for span in processed_spans}

    # 遍历序列构建注意力掩码
    for pos in range(seq_len):
        orig_idx, is_duplicate = index_mapping[pos]
        token = processed_inputs[pos]
        
        # 处理特殊标记
        if token == "<s>":
            attention_mask[pos, pos] = 1  # 句首仅关注自身
            active_tokens = {pos}         # 重置活跃集合
            continue
            
        if token == "</s>":
            continue  # 句尾标记不参与注意力
            
        # 处理首次关闭标记（组合注意力）
        if pos in close_positions:
            span = close_to_span[pos]
            # 设置对目标索引的注意力
            for target_idx in attention_indices[span["end"]]:
                attention_mask[pos, target_idx] = 1
            
            # 更新活跃集合
            active_tokens.add(pos)  # 添加当前关闭标记
            
            # 移除已关闭的起始标记
            if span["start"] in active_tokens:
                active_tokens.remove(span["start"])
            
            # 移除所有子跨度的结束标记
            for child in span["children"]:
                if child["end"] in active_tokens:
                    active_tokens.remove(child["end"])
            
            # 移除跨度内未被任何子跨度包含的原始终结符
            for orig_pos in range(span["orig_start"] + 1, span["orig_end"]):
                orig_token = action_sequence[orig_pos]
                # 跳过非终结符
                if orig_token.startswith("(") or orig_token.endswith(")"):
                    continue
                
                # 检查是否在子跨度内
                in_child_span = any(
                    child["orig_start"] < orig_pos < child["orig_end"]
                    for child in span["children"]
                )
                
                if not in_child_span:
                    processed_idx = original_to_first_occurrence[orig_pos]
                    if processed_idx in active_tokens:
                        active_tokens.remove(processed_idx)
            continue
        
        # 处理重复关闭标记（栈注意力）
        if pos in duplicate_positions:
            active_tokens.add(pos)
            for active_idx in active_tokens:
                attention_mask[pos, active_idx] = 1
            active_tokens.remove(pos)  # 立即移除
            continue
        
        # 处理开放标记（栈注意力）
        if pos in open_positions and not is_duplicate:
            active_tokens.add(pos)
            for active_idx in active_tokens:
                attention_mask[pos, active_idx] = 1
            continue
        
        # 处理终结符（栈注意力）
        if not (token.startswith("(") or token.endswith(")")):
            active_tokens.add(pos)
            for active_idx in active_tokens:
                attention_mask[pos, active_idx] = 1
            continue

    # 构建位置ID（基于深度
    position_ids = [0] * seq_len
    depth_stack = []  # 存储当前开放的非终结符类型
    
    for pos in range(seq_len):
        token = processed_inputs[pos]
        
        if token in ("<s>", "</s>"):
            position_ids[pos] = 0  # 特殊标记深度为0
            
        elif pos in close_positions:  # 首次关闭标记
            if not depth_stack:
                raise InvalidTreeError(f"位置 {pos} 处栈不平衡")
            depth_stack.pop()  # 关闭当前层级
            position_ids[pos] = len(depth_stack)
            
        elif pos in duplicate_positions:  # 重复关闭标记
            position_ids[pos] = len(depth_stack)  # 深度不变
            
        elif pos in open_positions:  # 开放标记
            position_ids[pos] = len(depth_stack)
            depth_stack.append(token[1:])  # 移除开括号并压栈
            
        else:  # 终结符
            position_ids[pos] = len(depth_stack)

    # 构建处理后的输出序列
    processed_outputs = []
    for pos, token in enumerate(processed_inputs):
        _, is_duplicate = index_mapping[pos]
        
        # 仅保留首次出现的关闭标记，重复关闭标记用<pad>替换
        if token.endswith(")") and is_duplicate:
            processed_outputs.append("<pad>")
        else:
            processed_outputs.append(token)
    
    return {
        "inputs": processed_inputs,
        "labels": processed_outputs,
        "position_ids": position_ids,
        "attention_mask": attention_mask
    }


def get_trainer(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    train_dataset: Dataset
) -> Trainer:
    """
    Question:
        Create a Trainer object for the model. The Trainer is used to train the model on the dataset.
        Select the appropriate training arguments for the Trainer. For example, setting the proper learning rate,
        batch size, optimizer, learning rate scheduler, number of epochs, etc. would be a good idea.

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use for the model.
        model (PreTrainedModel): The model to train.
        train_dataset (Dataset): The dataset to train on.

    Returns:
        trainer (Trainer): The Trainer object for the model.

    Example:
        >>> trainer = get_trainer(tokenizer, model, train_dataset)
        >>> trainer.train()
        >>> trainer.evaluate(train_dataset)
        {'eval_loss': 2.1234, ...}
    """

    def data_collator(features):
        """
        Data collator is to aggregate the features into a batch. You'll find it helpful when creating the Trainer.
        We simply pad the sequences but deal with attention mask seperately.
        """
        max_length = max([len(f["input_ids"]) for f in features])
        batch = {
            "input_ids": [],
            "labels": [],
            "position_ids": [],
            "attention_mask": [],
        }
        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]
            position_ids = f["position_ids"]
            attention_mask = f["attention_mask"]
            seq_len = len(input_ids)

            input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
            labels += [-100] * (max_length - len(labels))
            position_ids += [0] * (max_length - len(position_ids))
            attention_mask = F.pad(torch.tensor(attention_mask), [0, max_length - seq_len, 0, max_length - seq_len])

            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
            batch["position_ids"].append(position_ids)
            batch["attention_mask"].append(attention_mask)

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.stack(batch["attention_mask"])

        return batch

    """YOUR CODE HERE"""
    # util.raiseNotDefined()
    model_training_config = TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=1e-3,
        weight_decay=0.01,
        do_eval=False
    )

    trainer = Trainer(
        model=model,
        args=model_training_config,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        data_collator=data_collator,
    )

    return trainer


def main():
    """This function trains a Transformer Grammar model based on GPT2 for the task of generative transition-based parsing."""
 
    ## Load the dataset from disk
    dataset = load_dataset("text", data_files="data/corpus.cc", split="train")


    ## Build the word tokenizer
    # Initialize tokenizer with special tokens
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))

    # Use the whitespace pre-tokenizer to split on whitespace
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Build the vocabulary using WordLevelTrainer
    trainer = WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
    tokenizer.train_from_iterator(dataset["text"], trainer=trainer)

    # Set the post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )

    # Convert to PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>'})


    ## Preprocess the dataset
    def tokenize_function(example):
        tokenized = tokenizer.tokenize(example["text"], add_special_tokens=True)
        return {"actions": tokenized}

    def convert_function(examples):
        input_ids = tokenizer(examples["inputs"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        labels = tokenizer(examples["labels"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        labels = [[(idx if idx != tokenizer.pad_token_id else -100) for idx in sent] for sent in labels]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": examples["position_ids"],
            "attention_mask": [[mask] for mask in examples["attention_mask"]],
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"], load_from_cache_file=False)
    mapped_dataset = tokenized_dataset.map(mapping_function, batched=False, remove_columns=["actions"], load_from_cache_file=False)
    converted_dataset = mapped_dataset.map(convert_function, batched=True, remove_columns=["inputs"], load_from_cache_file=False)


    # Load the model
    # TODO: use GPT2 instead of GPTNeo when transformers 4.52.0 is released
    # We use GPTNeo here since the implementation of GPT2 has a bug and the fix has not been released yet.
    # GPTNeo is similar to GPT2 except that it uses local attention. We have disabled local attention in the config.
    config = GPTNeoConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_layers=6,
        num_heads=8,
        attention_types=[[["global"], 6]],
        activation_function="relu",
    )
    model = GPTNeoForCausalLM(config)


    # Training
    trainer = get_trainer(tokenizer, model, converted_dataset)
    trainer.train()
    metrics = trainer.evaluate(converted_dataset)

    print(metrics)


if __name__ == "__main__":
    main()
