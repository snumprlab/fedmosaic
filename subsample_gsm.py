import json
import random

def reservoir_sample_jsonl(input_file, output_file, sample_size, seed=None):
    """
    Perform reservoir sampling on a JSONL file to produce a smaller subset of lines.
    
    :param input_file: Path to the original JSONL file.
    :param output_file: Path to the resulting subsampled JSONL file.
    :param sample_size: Number of lines (records) to sample.
    :param seed: An optional integer to seed the random generator for reproducibility.
    """
    # if seed is not None:
    random.seed(42)

    # Initialize an empty list (the "reservoir")
    reservoir = []
    n = 0  # Tracks the total number of lines seen so far

    # Read lines one by one
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            # If we haven't filled the reservoir yet, just add the new line
            if n < sample_size:
                reservoir.append(line)
            else:
                # Once the reservoir is full, decide if we should replace an element
                # Choose a random index between 0 and n (inclusive)
                r = random.randint(0, n)
                if r < sample_size:
                    reservoir[r] = line
            n += 1

    # Write the final reservoir lines to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in reservoir:
            outfile.write(line)


if __name__ == "__main__":
    # Example usage:
    input_jsonl = "dataset/gsm8k/gsm8k_test.jsonl"
    output_jsonl = "dataset/gsm8k/gsm8k_test_subsampled.jsonl"
    sample_size = 200  # Number of lines you'd like to keep

    reservoir_sample_jsonl(input_jsonl, output_jsonl, sample_size)
    print(f"Created subsampled JSONL file with {sample_size} lines at {output_jsonl}")
