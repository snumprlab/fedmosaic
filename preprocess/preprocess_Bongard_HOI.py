import random
import os
import json
import jsonlines
from collections import defaultdict

random.seed(42)

num_per_set = 3

prompts = f'''Given {num_per_set} "positive" images and {num_per_set} "negative" images, where both "positive" and "negative" images share a "common" object, and only "positive" images share a "common" action whereas "negative" images have different actions compared to the "positive" images, the "common" action is exclusively depicted by the "positive" images. Your job is to find the "common" action within the "positive" images. You must choose your answer from the Choice List.'''

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
    
    for type_name, items in type_groups.items():
        # Randomly select items if there are more than samples_per_type
        selected_items = random.sample(items, min(samples_per_type, len(items)))
        
        for IIDEX, item in enumerate(selected_items):
            positive_imgfiles = item['image_files'][:7]  # Only use the first 6 positive images
            negative_imgfiles = item['image_files'][7:]  # Only use the first 6 negative images
            
            # Randomly select 3 positive and 3 negative images
            selected_positive = random.sample(positive_imgfiles, num_per_set)
            selected_negative = random.sample(negative_imgfiles, num_per_set)
            
            selected_imgfiles = ["dataset/Bongard-HOI/"+path for path in selected_positive + selected_negative]
            
            # Get the action classes
            action_classes = item['action_class']
            gt_action = action_classes[0]  # The first action is the ground truth for positive images
            
            # Create the choice list with unique elements
            choice_list = [gt_action]
            other_actions = list(set(action_classes[num_per_set:]) - set(choice_list))  # Exclude the positive actions and gt_action
            num_additional_choices = min(3, len(other_actions))
            choice_list.extend(random.sample(other_actions, num_additional_choices))
            # random.shuffle(choice_list)
            choice_list = sorted(choice_list)
            # Structure for LLaVA JSON
            json_data = {
                "id": f"{type_name}-{IIDEX}",
                "image": selected_imgfiles,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Positive: {'<image>'*num_per_set}\nNegative: {'<image>'*num_per_set}\n{prompts}\nChoice List: [{', '.join(choice_list)}]"
                    },
                    { 
                        "from": "gpt",
                        "value": gt_action
                    }
                ]
            }
            json_data_list.append(json_data)
    
    # Shuffle the final list to mix types
    random.shuffle(json_data_list)
    
    # Trim to max_samples if necessary
    json_data_list = json_data_list[:max_samples]
    
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, f'dataset-0.json')
    print(f"Total samples: {len(json_data_list)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)

# Usage example
output_folder = 'dataset/Bongard-HOI'

save_dataset('Bongard-HOI', output_folder, 'test', max_samples=2000)
save_dataset('Bongard-HOI', output_folder, 'train', max_samples=10000)