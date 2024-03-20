import pdb

from aac_metrics import evaluate
import json
import string

def get_reference(file_path):
    image_captions = {}

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        # Read the lines from the file
        lines = file.readlines()[1:]

        # Iterate through each line and update the dictionary
        for line in lines:
            parts = line.strip().split('.jpg,')
            image_file = parts[0].replace(".jpg", "")
            caption = parts[1].lower().strip()
            caption = caption.translate(str.maketrans('', '', string.punctuation)).strip()

            # Check if the image file is already in the dictionary
            if image_file in image_captions:
                # If yes, append the new caption to the existing list
                image_captions[image_file].append(caption)
            else:
                # If no, create a new list with the current caption
                image_captions[image_file] = [caption]
    return image_captions

def evaluate_result(image_captions, transcript_ars):
    candidates: list[str] = []
    mult_references: list[list[str]] = []
    for key, value in transcript_ars.items():
        candidates.append(value)
        mult_references.append(image_captions[key])
    score, _ = evaluate(candidates, mult_references)
    return score

if __name__ == "__main__":
    reference_path = "/home/ldap-users/Share/Corpora/Spoken_Image/Flickr8k_SAS/Data_for_SAS/captions.txt"
    image_captions = get_reference(reference_path)
    transcript_ars_path = "/home/ldap-users/s2220411/Code/new_explore_tts/BLIP/output/auditory_feedback_v9/auditory_feedback_ftlm_fs_2layer_cosine_MSEspeech_synthesis_last_hidden_state_fs/test_inference_50_tacotron2/transcript_ars.txt"
    transcript_ars = {}
    print(transcript_ars_path)
    with open(transcript_ars_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split('|')
            transcript_ars[parts[0]] = parts[1]

    score = evaluate_result(image_captions, transcript_ars)
    print(score)