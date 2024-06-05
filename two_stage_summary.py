import collections
import os
import re
import time
import ast
import openai
import pandas as pd
from multiprocessing.pool import Pool
import argparse
import base64
import json, random
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from tqdm import tqdm

openai.api_key = "You api_key"
openai.api_base = "You api_base"

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

with open('./example_summary.txt','r') as ex:
    example=ex.read()

systerm='''
You're a visual summary expert. You can accurately make a [SUMMARY] based on [CAPTION], where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.
'''
incontext_prompt='''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each line represents five captions of a 4s video clip, each caption is separated by a semicolon, with a total of 45 lines describing 180 seconds of video. Aat the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
[SUMMARY]: Based on the CAPTIONS of these video clips, you need to summarise them into an overall description of the video, in chronological order.
I will give you an example as follow:
<Example>
{example}
Now, you should make a [SUMMARY] based on the [CAPTION] below. You SHOULD follow the format of example.
[CAPTION]
{caption}
[SUMMARY]
'''

def gpt4(prompt):
    result = openai.ChatCompletion.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o",
    )
    return result.choices[0].message.content


def llm_inference(output_dir=None, part=None):

    for file in tqdm(part):
        uid = file[:-5]

        try:
            with open('./LaViLa_cap5/'+uid+'.json','r') as f:
                captions=json.load(f)
            caps=''
            for c in captions:
                caps+=c['Caption']+"\n"
            instruction=str(uid)+"\n"+systerm + "\n" + incontext_prompt.format(example=example, caption=caps)
            response = gpt4(instruction)

            with open(f"{output_dir}/{uid}.txt", "w") as f:
                f.write(response)
            
        except Exception as e:
            print(e)
            print(f"Error occurs when processing this query: {uid}", flush=True)
            traceback.print_exc()
            break


def main():
    qa_list=os.listdir('./LaViLa_cap5/')

    output_dir = './summary/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    exist_files = os.listdir(output_dir)
    qa_list = [item for item in qa_list if item.replace('json','txt') not in exist_files]
    print(len(qa_list))

    num_tasks = 10

    queries = {}
    with open('questions.json','r') as f1:
        data=json.load(f1)
    for item in data:
        queries[item['q_uid']] = item

    for i in range(100):
        
        try:
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in qa_list if f.replace('json','txt') not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(output_dir, part) for part in all_parts]

            with Pool(10) as pool:
                pool.starmap(llm_inference, task_args)
                
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()