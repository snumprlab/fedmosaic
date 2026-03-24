import json
import os
import random
import openai
from openai import OpenAI
import time
import jsonlines

NUM_SECONDS_TO_SLEEP = 0.01

client = OpenAI(
    # This is the default and can be omitted
    # api_key=os.environ.get("OPENAI_API_KEY"),
    api_key=""
)

random.seed(42)

output_folder = 'dataset/VizWiz-Original'
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

############################## first make annotation about subtask using gpt#########

def get_eval(content: str, max_tokens: int):
    user_content = [
                        {"type":"text", "text":content['text']},
                    ]

    while True:
        try:
            '''
            
            '''
            response = client.chat.completions.create(
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant to categorize given question.'
                },{
                    'role': 'user',
                    'content': user_content
                }],
                model="gpt-4o-mini-2024-07-18", # TODO: Choose gpt version
                temperature=0.0,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        time.sleep(NUM_SECONDS_TO_SLEEP)
    return response.choices[0].message.content

def parse_score(review):
    try:
        score = review.split('[[')
        assert len(score) == 2
        score = score[-1].split(']]')[0]
        # return int(score)
        return score

    except Exception as e:
        print(e)
        print('error', review)
        return -1

rule = {"role": "Assistant", 
        "prompt": 
            "Please act as a classifier that classifies the given question asking about an image into one of the 4 question types displayed below. The 4 question types are:\n"
            "1. Object Identification Questions: Questions asking to identify an object, such as \"What is this?\", \"What is this item?\", or \"What is this product?\"\n"
            "2. Color Identification Questions: Questions specifically asking about the color of an object, like \"What color is this?\", \"What color are my shoes?\", or \"What color is this shirt?\"\n"
            "3. Food and Drink Identification: Questions related to identifying food or beverages, such as \"What is in this can?\", \"What flavor is this?\", or \"What type of drink is this?\"\n"
            "4. Brand and Label Identification: Questions asking to identify the brand or text on a label, such as \"What brand is this?\", \"What does this label say?\", or \"What is written on this bottle?\"\n"
            "5. Size or Quantity Questions: Questions asking for quantities, measurements, or sizes, such as \"What size is this shirt?\", \"How much is this?\", or \"How many calories are in this drink?\"\n"
            "6. Instruction or Direction Questions: Questions asking for instructions, such as \"What are the instructions for this product?\", or \"What does this sign say?\"\n"
            "7. Location or Setting Questions: Questions asking for details about a location, setting, or environment, such as \"Where is this?\", \"What room is this?\", or \"Is this the backyard?\"\n"
            "8. Product Type or Category: Questions asking to categorize or identify the type of product, like \"What kind of soup is this?\", \"What type of cereal is this?\", or \"What type of coffee is this?\"\n"
            "9. Time or Date Related Questions: Questions asking about time, such as \"What is the expiration date?\", \"When does this expire?\", or \"What time is it?\"\n"
            "10. Device or Equipment Identification: Questions related to identifying devices or equipment, such as \"What kind of TV is this?\", \"What is this computer screen showing?\", or \"What is this exercise machine?\"\n"
            "Begin your evaluation by providing a short explanation. Be as objective as possible."
            " After providing your explanation, you must return the type of the question among 1 to 10 in a single integer by strictly following this format:\"[[type]]\", for example: \"Question type: [[2]]\"."
        }


def categorize_question_type(split_name):

    original_datalist = json.load(open(os.path.join(output_folder, f'{split_name}.json'),'r'))
    new_datalist = []

    review_file = open(f'vizwiz_type_explanation2.jsonl', 'a')
    # gpt_outputs = []
    # with jsonlines.open(f'vizwiz_type_explanation3.jsonl') as f:
    #     for line in f.iter():
    #         gpt_outputs.append(line)
    
    # type3_cnt = 0
    for item in original_datalist:
        question = item['question']
        
        content = {
            "text": f"[Instruction]\n{rule['prompt']}\n\n"
                    f"[Question]\n{question}\n"
        }
        review = get_eval(content, 512)
        question_type = parse_score(review)
        if int(question_type) == -1:
            continue
        else:
            item['question_type'] = int(question_type)
        
        review_file.write(json.dumps({"question": question, "gpt_response":review}) + "\n")
        # review_file.flush()
        
        new_datalist.append(item)

    with open(os.path.join(output_folder, f'{split_name}_split.json'), 'w') as fp:
        json.dump(new_datalist, fp, indent=4)

# categorize_question_type('val')
# validation cost: $0.37 for gpt4o-mini
# breakpoint()
categorize_question_type('train')
breakpoint()
###########################################################################################