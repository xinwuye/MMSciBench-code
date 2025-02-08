import os
import anthropic
import base64
import imghdr
from tqdm import tqdm
import re
import pandas as pd
import time

# Set the API key for Anthropic
os.environ['ANTHROPIC_API_KEY'] = 'api-key'


def claude_multi_choice_has_img(multi_choice_has_img):
    multi_choice_has_img = multi_choice_has_img.copy()
    client = anthropic.Anthropic()

    # Updated system prompt to specify format without enforcing explanation
    system_prompt = (
        'As an AI tutor, answer the provided question and conclude your response by stating the selected choice(s).'
    )

    # Prepare messages to send for each question
    messages_to_send = []

    for _, row in multi_choice_has_img.iterrows():
        message_content = []

        # Add the text message
        message_content.append({"type": "text", "text": row['msg']})
        
        # Check if there are images and append text if needed
        if row['img']:
            message_content.append({"type": "text", "text": "相关图片："})

            # Add each image as a base64 encoded message
            for img_path in row['img']:
                with open(img_path, "rb") as image_file:
                    image_data = image_file.read()
                    
                    # Detect the image type (e.g., 'jpeg', 'png')
                    image_type = imghdr.what(None, image_data)
                    
                    # Map image types to media types
                    media_type = f"image/{image_type}" if image_type else "image/png"  # default to "image/png" if undetected
                    
                    # Encode the image data to base64
                    base64_string = base64.standard_b64encode(image_data).decode('utf-8')
                    
                    message_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_string
                        }
                    })
        
        message_content.append({
            'type': 'text', 'text': "Notice, you MUST answer in English. Let's solve it step by step. "
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
        success = False
        while not success:
            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=8192,
                    temperature=0,
                    system=system_prompt,
                    messages=[message]
                )
                success = True 
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)
        
        full_response = response.content[0].text
        responses.append(full_response)

    # Store the extracted answer and full response directly in multi_choice_has_img
    multi_choice_has_img['response'] = responses

    return multi_choice_has_img


def claude_multi_choice_no_img(multi_choice_no_img):
    multi_choice_no_img = multi_choice_no_img.copy()
    client = anthropic.Anthropic()

    # Updated system prompt to specify format without enforcing explanation
    system_prompt = (
        'As an AI tutor, answer the provided question and conclude your response by stating the selected choice(s).'
    )

    # Prepare messages to send for each question
    messages_to_send = []

    for _, row in multi_choice_no_img.iterrows():
        message_content = []

        # Add the text message
        message_content.append({"type": "text", "text": row['msg']})

        message_content.append({
            'type': 'text', 'text': "Notice, you MUST answer in English. Let's solve it step by step. "
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
        success = False
        while not success:
            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=8192,
                    temperature=0,
                    system=system_prompt,
                    messages=[message]
                )
                success = True 
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)
        
        full_response = response.content[0].text
        responses.append(full_response)

    # Store the extracted answer and full response directly in multi_choice_no_img
    multi_choice_no_img['response'] = responses
    
    return multi_choice_no_img


def claude_qa_has_img(qa_has_img):
    qa_has_img = qa_has_img.copy()
    client = anthropic.Anthropic()

    # Updated system prompt to specify format without enforcing explanation
    system_prompt = (
        "As an AI tutor, you should answer the provided question."
    )

    # Prepare messages to send for each question
    messages_to_send = []

    for _, row in qa_has_img.iterrows():
        message_content = []

        # Add the text message
        message_content.append({"type": "text", "text": row['msg']})
        
        # Check if there are images and append text if needed
        if row['content_img']:
            message_content.append({"type": "text", "text": "相关图片："})

            # Add each image as a base64 encoded message
            for img_path in row['content_img']:
                with open(img_path, "rb") as image_file:
                    image_data = image_file.read()
                    
                    # Detect the image type (e.g., 'jpeg', 'png')
                    image_type = imghdr.what(None, image_data)
                    
                    # Map image types to media types
                    media_type = f"image/{image_type}" if image_type else "image/png"  # default to "image/png" if undetected
                    
                    # Encode the image data to base64
                    base64_string = base64.standard_b64encode(image_data).decode('utf-8')
                    
                    message_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_string
                        }
                    })

        message_content.append({
            'type': 'text', 'text': "Notice, you MUST answer in English. Let's solve it step by step. "
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
        success = False
        while not success:
            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=8192,
                    temperature=0,
                    system=system_prompt,
                    messages=[message]
                )
                success = True 
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)
        
        full_response = response.content[0].text
        responses.append(full_response)

    qa_has_img['response'] = responses
    return qa_has_img


def claude_qa_no_img(qa_no_img):
    qa_no_img = qa_no_img.copy()
    client = anthropic.Anthropic()

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
            'type': 'text', 'text': "Notice, you MUST answer in English. Let's solve it step by step. "
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
        success = False
        while not success:
            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=8192,
                    temperature=0,
                    system=system_prompt,
                    messages=[message]
                )
                success = True 
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)
        
        full_response = response.content[0].text
        responses.append(full_response)

    qa_no_img['response'] = responses
    return qa_no_img