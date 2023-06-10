import argparse
import random
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
# import copy
import torch
import numpy as np
import json
import nltk
import time
# import time
# from torch import nn
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoTokenizer,GPTJForCausalLM
# from sentence_transformers import SentenceTransformer
# from datasets import load_dataset
# from sklearn.metrics import f1_score
from MetaICL.metaicl.data import MetaICLData
from MetaICL.metaicl.model import MetaICLModel
# from collections import defaultdict
from get_task import get_task
from utils import calculate_sentence_transformer_embedding,codex_execution,expand_to_aliases
from two_steps import selective_annotation,prompt_retrieval

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', required=True,type=str)
parser.add_argument('--selective_annotation_method', required=True,type=str)
parser.add_argument('--model_cache_dir', required=True,type=str)
parser.add_argument('--data_cache_dir', required=True,type=str)
parser.add_argument('--output_dir', required=True,type=str)
parser.add_argument('--prompt_dir', required=True, type=str)
parser.add_argument('--prompt_part', default=0, type=int)
parser.add_argument('--prompt_start', default=89, type=int)
parser.add_argument('--prompt_num', default=12, type=int)
parser.add_argument('--model_key', type=str)
parser.add_argument('--prompt_retrieval_method', default='similar',type=str)
parser.add_argument('--model_name', default='EleutherAI/gpt-j-6B',type=str)
parser.add_argument('--embedding_model', default='sentence-transformers/paraphrase-mpnet-base-v2',type=str)
parser.add_argument('--annotation_size', default=100,type=int)
parser.add_argument('--seed', default=0,type=int)
parser.add_argument('--batch_size', default=10,type=int)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

if __name__=='__main__':
    set_seed(args.seed)

    if args.task_name in ['mnli','rte','sst5','mrpc','dbpedia_14','hellaswag','xsum','nq','diffusiondb']:
        if args.task_name=='xsum':
            tokenizer_gpt = AutoTokenizer.from_pretrained(args.model_name,cache_dir=args.model_cache_dir)
            inference_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=args.model_cache_dir)
            inference_model.cuda()
            inference_model.eval()
            data_module = None
            return_string = True
            device = torch.device('cuda')
            single_input_len = None
            maximum_input_len = 1900
        elif args.task_name=='nq':
            maximum_input_len = 3800
            return_string = True
            single_input_len = None
            inference_model = None
            data_module = None
            tokenizer_gpt = None
            model_keys = args.model_key.split('##')
        else:
            data_module = MetaICLData(method="direct", max_length=1024, max_length_per_example=256)
            inference_model = MetaICLModel(args=args)
            inference_model.load()
            inference_model.cuda()
            inference_model.eval()
            tokenizer_gpt = None
            return_string = False
            single_input_len = 250
            maximum_input_len = 1000

        output_dir = args.output_dir
        bar = tqdm(range(args.prompt_start, args.prompt_start + args.prompt_num), desc=f'annotation for prompts: ')
        for prompt_id in range(args.prompt_start, args.prompt_start + args.prompt_num):
            args.prompt_part = prompt_id
            args.output_dir = os.path.join(output_dir, f'{args.prompt_part}')
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir,exist_ok=True)
            train_examples,eval_examples,train_text_to_encode,eval_text_to_encode,format_example,label_map = get_task(args=args)
            total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=train_text_to_encode, args=args)
            total_eval_embeds = calculate_sentence_transformer_embedding(text_to_encode=eval_text_to_encode, args=args)

            if os.path.isfile(os.path.join(args.output_dir,'first_phase_selected_indices.json')):
                with open(os.path.join(args.output_dir,'first_phase_selected_indices.json')) as f:
                    first_phase_selected_indices = json.load(f)
            else:
                first_phase_selected_indices = selective_annotation(embeddings=total_train_embeds,
                                                                    train_examples=train_examples,
                                                                    return_string=return_string,
                                                                    format_example=format_example,
                                                                    maximum_input_len=maximum_input_len,
                                                                    label_map=label_map,
                                                                    single_context_example_len=single_input_len,
                                                                    inference_model=inference_model,
                                                                    inference_data_module=data_module,
                                                                    tokenizer_gpt=tokenizer_gpt,
                                                                    args=args)
                with open(os.path.join(args.output_dir,'first_phase_selected_indices.json'),'w') as f:
                    json.dump(first_phase_selected_indices,f,indent=4)
            
            bar.update(1)
            
