import google.generativeai as genai
import time
import re
from tqdm import tqdm

# Authenticate with Gemini API
genai.configure(api_key='api-key')


def gemini_multi_choice_has_img(multi_choice_has_img):
    multi_choice_has_img = multi_choice_has_img.copy()
    # Define the system instruction as a system prompt
    system_instruction = 'As an AI tutor, answer the provided question and conclude your response by stating the selected choice(s).'

    generation_config = {
    "temperature": 0,
    # "max_output_tokens": 1000,
    "response_mime_type": "text/plain",
    }

    # Initialize the Gemini model with the system instruction
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        system_instruction=system_instruction,
        generation_config=generation_config,
    )

    # Prepare the questions for API calls
    responses = []
    results = []

    for _, row in tqdm(multi_choice_has_img.iterrows()):
        question_content = row['msg']
        content_parts = [question_content]

        # If there are images, upload them and append to the content
        if row.get('img'):
            content_parts.append("\n\n相关图片：\n")
            for img_path in row['img']:
                success = False
                while not success:
                    try:
                        # Upload the image file to Gemini and get a reference
                        uploaded_file = genai.upload_file(img_path)
                        content_parts.append(uploaded_file)
                        success = True
                    except Exception as e:
                        print(f"Error uploading image {img_path}: {e}")
                        time.sleep(2)

        content_parts.append("Notice, you MUST answer in English. Let's solve it step by step. ")

        # Combine all content parts
        # full_content = "".join(content_parts)
        full_content = content_parts

        success = False
        while not success:
            try:
                # Call the Gemini model's generate_content method
                result = model.generate_content(full_content)
                response_text = result.text.strip()
                success = True  # Exit loop if successful
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)

        responses.append(response_text)

    # Append responses and results to the DataFrame
    multi_choice_has_img['response'] = responses

    return multi_choice_has_img


def gemini_multi_choice_no_img(multi_choice_no_img):
    multi_choice_no_img = multi_choice_no_img.copy()
    # Define the system instruction as a system prompt
    system_instruction = 'As an AI tutor, answer the provided question and conclude your response by stating the selected choice(s).'

    generation_config = {
    "temperature": 0,
    # "max_output_tokens": 1000,
    "response_mime_type": "text/plain",
    }

    # Initialize the Gemini model with the system instruction
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        system_instruction=system_instruction,
        generation_config=generation_config,
    )

    # Prepare the questions for API calls
    responses = []
    results = []

    for _, row in tqdm(multi_choice_no_img.iterrows()):
        question_content = row['msg']
        content_parts = [question_content]

        content_parts.append("Notice, you MUST answer in English. Let's solve it step by step. ")
        
        success = False
        while not success:
            try:
                # Call the Gemini model's generate_content method
                result = model.generate_content(content_parts)
                response_text = result.text.strip()
                success = True  # Exit loop if successful
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)

        responses.append(response_text)

    # Append responses and results to the DataFrame
    multi_choice_no_img['response'] = responses
    
    return multi_choice_no_img


def gemini_qa_has_img(qa_has_img):
    qa_has_img = qa_has_img.copy()
    # Define the system instruction as a system prompt
    system_instruction = (
        "As an AI tutor, you should answer the provided question."
    )

    generation_config = {
        "temperature": 0,
        # "max_output_tokens": 1000,
        "response_mime_type": "text/plain",
    }

    # Initialize the Gemini model with the system instruction
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        system_instruction=system_instruction,
        generation_config=generation_config,
    )

    # Prepare the questions for API calls
    responses = []

    for _, row in tqdm(qa_has_img.iterrows()):
        question_content = row['msg']
        content_parts = [question_content]

        # If there are images, upload them and append to the content
        if row.get('content_img'):
            content_parts.append("\n\n相关图片：\n")
            for img_path in row['content_img']:
                success = False
                while not success:
                    try:
                        # Upload the image file to Gemini and get a reference
                        uploaded_file = genai.upload_file(img_path)
                        content_parts.append(uploaded_file)
                        success = True
                    except Exception as e:
                        print(f"Error uploading image {img_path}: {e}")
                        time.sleep(2)

        content_parts.append("Notice, you MUST answer in English. Let's solve it step by step. ")

        # Combine all content parts into the input for the model
        full_content = content_parts

        success = False
        while not success:
            try:
                # Call the Gemini model's generate_content method
                result = model.generate_content(full_content)
                response_text = result.text.strip()
                success = True  # Exit loop if successful
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)

        responses.append(response_text)

    # Append responses to the DataFrame
    qa_has_img['response'] = responses

    return qa_has_img


def gemini_qa_no_img(qa_no_img):
    qa_no_img = qa_no_img.copy()
    # Define the system instruction as a system prompt
    system_instruction = (
        "As an AI tutor, you should answer the provided question."
    )

    generation_config = {
        "temperature": 0,
        # "max_output_tokens": 1000,
        "response_mime_type": "text/plain",
    }

    # Initialize the Gemini model with the system instruction
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        system_instruction=system_instruction,
        generation_config=generation_config,
    )

    # Prepare the questions for API calls
    responses = []

    for _, row in tqdm(qa_no_img.iterrows()):
        question_content = row['msg']
        content_parts = [question_content]

        content_parts.append("Notice, you MUST answer in English. Let's solve it step by step. ")

        success = False
        while not success:
            try:
                # Call the Gemini model's generate_content method
                result = model.generate_content(content_parts)
                response_text = result.text.strip()
                success = True  # Exit loop if successful
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)

        responses.append(response_text)

    # Append responses to the DataFrame
    qa_no_img['response'] = responses

    return qa_no_img