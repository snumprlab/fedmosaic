import random
import os
import json
from collections import defaultdict

random.seed(42)

prompts = f'''Given a set of images where all share a common object, identify the one image that displays a different action while the rest share the same action. You must choose your answer from the Choice List.'''

def save_dataset(dataset_name, output_folder, subset_name, max_samples):
    subset_folder = os.path.join(output_folder, subset_name)
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)
        
    with open(f"{output_folder}/{subset_name}.json") as fp:
        datalist = json.load(fp)
    
    # Group items by type
    type_groups = defaultdict(list)
    for item in datalist:
        type_groups[item['type']].append(item)
    
    # Calculate samples per type
    num_types = len(type_groups)
    samples_per_type = max_samples // num_types
    print(samples_per_type)
    json_data_list = []
    
    answers = ['Image A', 'Image B', 'Image C', 'Image D']
    
    for type_name, items in type_groups.items():
        # Randomly select items if there are more than samples_per_type
        selected_items = random.sample(items, min(samples_per_type, len(items)))
        
        for IIDEX, item in enumerate(selected_items):
            positive_imgfiles = item['image_files'][:7]  # Only use the first 6 positive images
            negative_imgfiles = item['image_files'][7:]  # Only use the first 6 negative images
            
            # Randomly select 3 positive and 3 negative images
            selected_positive = random.sample(positive_imgfiles, 3)
            selected_negative = random.sample(negative_imgfiles, 1)
            
            answer_index = random.randint(0, 3)
            
            selected_imgfiles = ["dataset/Bongard-HOI/"+path for path in selected_positive]
            selected_imgfiles.insert(answer_index, "dataset/Bongard-HOI/"+selected_negative[0])
            
            # Structure for LLaVA JSON
            json_data = {
                "id": f"{type_name}-{IIDEX}",
                "image": selected_imgfiles,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{prompts}\nImage A: <image>, Image B: <image>, Image C: <image>, Image D: <image>\nChoice list:[Image A, Image B, Image C, Image D]. Your answer is: "
                    },
                    { 
                        "from": "gpt",
                        "value": answers[answer_index]
                    }
                ]
            }
            json_data_list.append(json_data)
    
    # Shuffle the final list to mix types
    random.shuffle(json_data_list)
    
    # Trim to max_samples if necessary
    json_data_list = json_data_list[:max_samples]
    
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, f'dataset-2.json')
    print(f"Total samples: {len(json_data_list)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)

# Usage example
output_folder = 'dataset/Bongard-HOI'

save_dataset('Bongard-HOI', output_folder, 'test', max_samples=2000)
save_dataset('Bongard-HOI', output_folder, 'train', max_samples=10000)