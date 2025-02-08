import os
import base64
import imghdr
from tqdm import tqdm
import re
import pandas as pd
from openai import OpenAI
import openai
import time


# Function to convert image paths to base64-encoded format
def encode_image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        image_data = image_file.read()
        image_type = imghdr.what(None, image_data)
        base64_string = base64.b64encode(image_data).decode('utf-8')
        return base64_string, image_type



def openai_multi_choice_no_img(multi_choice_no_img, model_name, base_url, api_key):
    multi_choice_no_img = multi_choice_no_img.copy()
    client = OpenAI(
        base_url=base_url, 
        api_key=api_key 
    )

    # Updated system prompt to specify format without enforcing explanation
    system_prompt = 'As an AI tutor, answer the provided question and conclude your response by stating the selected choice(s).'

    # Prepare messages to send for each question
    messages_to_send = []

    for _, row in multi_choice_no_img.iterrows():
        message_content = []

        # Add the text message
        message_content.append({"type": "text", "text": row['msg']})

        message_content.append({
            'type': 'text', 'text': "Notice, you MUST answer in Chinese. Let's solve it step by step. "
        })

        # Construct the message for Anthropic's Claude
        messages_to_send.append({
            "role": "user",
            "content": message_content,
        })

    # Send the messages to Claude and process responses
    responses = []
    results = []
    for message in tqdm(messages_to_send):
        # Send the prompt to GPT-4o in a single message
        success = False
        while not success:
            try:
                response = client.chat.completions.create(
                    temperature=0.,
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        message
                    ],
                    max_tokens=1000,
                )
                success = True  # Exit the loop if the request is successful

            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)
        
        full_response = response.choices[0].message.content.strip()
        responses.append(full_response)

    # Store the extracted answer and full response directly in multi_choice_no_img
    multi_choice_no_img['response'] = responses
    return multi_choice_no_img




def openai_qa_no_img(qa_no_img, model_name, base_url, api_key):
    qa_no_img = qa_no_img.copy()
    client = OpenAI(
        base_url=base_url, 
        api_key=api_key 
    )

    # Updated system prompt to specify format without enforcing explanation
    system_prompt = (
        "As an AI tutor, you should answer the provided question."
    )

    # Prepare messages to send for each question
    messages_to_send = []

    for _, row in qa_no_img.iterrows():
        message_content = []

        # Add the text message
        message_content.append({"type": "text", "text": row['msg']})

        message_content.append({
            'type': 'text', 'text': "Notice, you MUST answer in Chinese. Let's solve it step by step. "
        })

        # Construct the message for Anthropic's Claude
        messages_to_send.append({
            "role": "user",
            "content": message_content,
        })

    # Send the messages to Claude and process responses
    responses = []
    results = []
    for message in tqdm(messages_to_send):
        # Send the prompt to GPT-4o in a single message
        success = False
        while not success:
            try:
                response = client.chat.completions.create(
                    temperature=0.,
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        message
                    ],
                    max_tokens=1000,
                )
                success = True  # Exit the loop if the request is successful
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)
        
        full_response = response.choices[0].message.content.strip()
        responses.append(full_response)

    qa_no_img['response'] = responses

    return qa_no_img





def qwen25math72b_multi_choice_no_img(data):
    return openai_multi_choice_no_img(
        data, 
        'tgi',
        "https://yvxaveobgc1yqym2.us-east-1.aws.endpoints.huggingface.cloud/v1/",
        'api-key'
    )


def qwen25math72b_qa_no_img(data):
    return openai_qa_no_img(
        data, 
        'tgi',
        "https://yvxaveobgc1yqym2.us-east-1.aws.endpoints.huggingface.cloud/v1/",
        'api-key'
    )


def deepseekmath7b_multi_choice_no_img(data):
    return openai_multi_choice_no_img(
        data, 
        'tgi',
        "https://iw9n1cvzow1w3xba.us-east-1.aws.endpoints.huggingface.cloud/v1/",
        'api-key'
    )


def deepseekmath7b_qa_no_img(data):
    return openai_qa_no_img(
        data, 
        'tgi',
        "https://iw9n1cvzow1w3xba.us-east-1.aws.endpoints.huggingface.cloud/v1/",
        'api-key'
    )