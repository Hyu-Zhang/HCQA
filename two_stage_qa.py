import collections
import os
import re
import time
import ast
import openai
import pandas as pd
from multiprocessing.pool import Pool
import base64
import json, random
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from tqdm import tqdm
import argparse

openai.api_key = "You api_key"
openai.api_base = "You api_base"
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

with open('./example_qa.txt','r') as ex:
    example=ex.read()

systerm='''
You are a visual question answering expert. You can choose the correct answer from five options of [OPTION] based on the [CAPTION], [SUMMARY], [QUESTION], and [REASON]. Where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.
'''
incontext_prompt='''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each line represents five captions of a 4s video clip, each caption is separated by a semicolon, with a total of 45 lines describing 180 seconds of video. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
[SUMMARY]: Based on the CAPTIONS of these video clips, an overall description of the video, in chronological order.
[QUESTION]: A question about video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [SUMMARY], reasoning step by step to get the answer. If [SUMMARY] doesn't have enough information, you need to get it from the [CAPTION].
I will give you some examples as follow:
{example}
Now, you should first make a [REASON] based on [QUESTION] and [SUMMARY], then give right number of [OPTION] as [ANSWER] . Additionally, you need to give me [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You SHOULD answer the question, even given a low confidence.
[CAPTION]
{caption}
[SUMMARY]
{summary}
[QUESTION]
{question}
[OPTION]
{option}
'''
response_prompt='''
YOU MUST output in the JSON format.
{
'REASON':[REASON],
'ANSWER':[ANSWER],
'CONFIDENCE': [CONFIDENCE]
}
'''

def gpt4(prompt):
    result = openai.ChatCompletion.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o",
        response_format={"type": "json_object"}
    )
    return result.choices[0].message.content


def llm_inference(queries, output_dir=None, part=None):

    for file in tqdm(part):
        uid = file[:-5]
        d = queries[uid]
        try:
            with open('./LaViLa_cap5/'+uid+'.json','r') as f:
                captions=json.load(f)
            caps=''
            for c in captions:
                caps+=c['Caption']+"\n"
            with open('./summary/'+uid+'.txt','r') as f1:
                sum=f1.read()
            opt=''
            que='question: '+d['question']
            opt+='option 0: '+d['option 0']+"\n"
            opt+='option 1: '+d['option 1']+"\n"
            opt+='option 2: '+d['option 2']+"\n"
            opt+='option 3: '+d['option 3']+"\n"
            opt+='option 4: '+d['option 4']+"\n"

            instruction=str(uid)+"\n"+systerm + "\n" + incontext_prompt.format(example=example, caption=caps, summary=sum, question=que, option=opt)+"\n"+response_prompt
            response = gpt4(instruction)

            response_dict = ast.literal_eval(response)
            with open(f"{output_dir}/{uid}.json", "w") as f:
                json.dump(response_dict, f)

        except openai.RateLimitError:
            print("Too many request: Sleeping 1s", flush=True)
            time.sleep(1)
        except Exception:
            print(f"Error occurs when processing this query: {uid}", flush=True)
            traceback.print_exc()
            break
        else:
            break

def main():
    qa_list=os.listdir('./LaViLa_cap5/')

    output_dir = './result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    exist_files = os.listdir(output_dir)
    qa_list = [item for item in qa_list if item not in exist_files]
    print(len(qa_list))

    num_tasks = 10

    queries = {}
    with open('questions.json','r') as f1:
        data=json.load(f1)
    for item in data:
        queries[item['q_uid']] = item

    while True:
        try:
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in qa_list if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(queries, output_dir, part) for part in all_parts]

            with Pool(10) as pool:
                pool.starmap(llm_inference, task_args)

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()