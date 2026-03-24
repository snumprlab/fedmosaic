import os
import re

def clean_checkpoints(directory, round_numbers):
    pattern = re.compile(r'(\d+)_client_model_round(\d+)\.pth')
    keep_files = set()
    
    # Identify files to keep
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            round_num = int(match.group(2))
            if round_num in round_numbers:
                keep_files.add(filename)
    #print(keep_files)
    
    # Delete files not in the keep list
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if 'trainer_state' not in file_path and os.path.isfile(file_path) and filename not in keep_files:
            os.remove(file_path)
            print(f"Deleted: {filename}")

# Example usage
directory = "client_states_feddat_scenario320_fedavg_iter100_round5_homo_3B"
num_rounds = 10
if num_rounds == 10:
    round_numbers = {5, 10, 15, 20, 25, 30, 35, 40}
else:
    round_numbers = {2, 5, 7, 10, 12, 15, 17, 20}
clean_checkpoints(directory, round_numbers)
