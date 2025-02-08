from openai import OpenAI
import openai
from tqdm import tqdm
import os
import re
import time
import base64
import imghdr
import pandas as pd

os.environ["OPENAI_API_KEY"] = "api-key"


# Function to convert image paths to base64-encoded format
def encode_image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        image_data = image_file.read()
        image_type = imghdr.what(None, image_data)
        base64_string = base64.b64encode(image_data).decode('utf-8')
        return base64_string, image_type



def gpt4o_judge4multi_choice(multi_choice):
    multi_choice = multi_choice.copy()
    client = openai.OpenAI()

    # System prompt tailored for multi-choice answer correctness checking
    system_prompt = '''
    你是一个助教助手，负责判断学生答案的选择是否与标准答案一致。
    '''

    # Prepare to store GPT-4o ratings
    final_answer_correctness = []

    for _, row in tqdm(multi_choice.iterrows(), total=multi_choice.shape[0]):
        # Prepare the content for evaluation
        user_content = [
            {"type": "text", "text": f"### 标准答案：\n{row['sol']}"},
            {"type": "text", "text": f"### 学生答案：\n{row['response']}"}
        ]

        requirement_prompt = '''
        标准答案只包含选项，而学生答案可能包含思考过程或解析。
        你需要从学生答案中提取具体选择项，并与标准答案进行比对。
        如果提取出的学生选择与标准答案完全一致，则回答“正确”，否则回答“错误”。
        判断结果以“正确”或“错误”的形式回答，不要提供任何其他信息。
        '''
        user_content.append({'type': 'text', 'text': requirement_prompt})

        # Send the prompt to GPT-4o in a single message
        success = False
        while not success:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=10,
                )
                # Extract the model's response content
                result = response.choices[0].message.content.strip()
                # Validate the response
                if result in {"正确", "错误"}:  # Check if the result is valid
                    final_answer_correctness.append(result)
                    success = True  # Exit the loop if successful
                else:
                    print(f"Invalid response: {result}. Retrying...")
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)

    # Add the correctness results as a new column in the DataFrame
    multi_choice['final_answer_correctness'] = final_answer_correctness

    return multi_choice


def gpt4o_judge4qa_has_img_explanation1(qa_has_img):
    qa_has_img = qa_has_img.copy()
    client = openai.OpenAI()

    # System prompt tailored for checking answer correctness with flexible expression matching
    system_prompt = '''
    你是一个助教助手，负责判断学生答案的结论是否与标准答案的结论表达同样的意思。
    '''

    # Filter questions that are not proof questions
    # qa_has_img_filtered = qa_has_img[qa_has_img['is_proof_question'] == 'NO']
    qa_has_img_filtered = qa_has_img

    # Prepare to store GPT-4o ratings
    final_answer_correctness = []
    results = []

    for _, row in tqdm(qa_has_img_filtered.iterrows(), total=qa_has_img_filtered.shape[0]):
        # Prepare the content list for evaluation
        user_content = [
            {"type": "text", "text": "请根据以下题目信息和提供的标准答案判断学生答案是否正确:"},
            {"type": "text", "text": f"### 问题：\n{row['msg']}"},
        ]

        # Add question images if they exist
        if row['content_img']:
            user_content.append({"type": "text", "text": "#### 相关图片："})
            for img_path in row['content_img']:
                base64_string, image_type = encode_image_to_base64(img_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_type};base64,{base64_string}"}
                })

        user_content.append({"type": "text", "text": f"\n### 标准答案：\n{row['ans']}"})
        
        # Add correct answer images if they exist
        if row['answer_img']:
            user_content.append({"type": "text", "text": "#### 标准答案相关图片："})
            for img_path in row['answer_img']:
                base64_string, image_type = encode_image_to_base64(img_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_type};base64,{base64_string}"}
                })

        user_content.append({"type": "text", "text": f"\n### 学生答案：\n{row['response']}"})

        requirement_prompt = '''
        请根据以上题目信息和标准答案，仅根据学生答案的最终结论或答案判断其是否正确，忽略过程的正确性。

        注意事项：
        1. 检查题目是否包含多个子问题：
        - 如果包含多个子问题，请逐一判断每个子问题的答案是否正确。只有当所有子问题的最终答案都正确时，整体答案才被视为正确。
        - 如果不包含子问题，则仅根据学生答案的最终结论或答案进行判断。
        2. 即使学生答案的表达方式与标准答案不同，只要最终结论或答案的意思相同，也应视为正确。可能的情况包括但不限于：
        - 学生答案和标准答案使用的语言不同，但意思相同。
        - 学生答案中的公式经过化简和变形后与标准答案的公式相同。
        - 学生答案采用了不同的表述形式，但语义相同。
        3. 请解释和分析学生答案与标准答案的最终结论或答案的异同之处。
        4. 如果由于学生答案不完整或缺失，或者学生答案没有按照题目要求给出结论，导致无法判断是否正确的，就判断为错误。
        5. 如果题目包含子问题，请为每个子问题提供“子问题X判断结果：正确”或“子问题X判断结果：错误”的形式。
        6. 最终判断结果应以“判断结果：正确”或“判断结果：错误”的形式给出。

        请按照以下格式回复：

        分析：
        [在此处填写详细的分析内容]

        [如果有子问题，则添加以下部分]
        子问题判断结果：
        子问题1判断结果：正确/错误
        子问题2判断结果：正确/错误
        ...
        子问题N判断结果：正确/错误

        判断结果：正确/错误
        '''
        user_content.append({'type': 'text', 'text': requirement_prompt})

        # Send the prompt to GPT-4o in a single message
        success = False
        while not success:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                )
                # Extract the model's response content
                result = response.choices[0].message.content.strip()

                # 无子问题时，只提取最终判断结果
                final_pattern = r'(?<!\d)判断结果：(正确|错误)'
                matches = re.findall(final_pattern, result)
                if matches:
                    # Select the last match from the list
                    final_result = matches[-1]
                    # Append the result to your existing lists
                    final_answer_correctness.append(final_result)
                    results.append(result)
                    # Indicate successful extraction
                    success = True  # 成功获取有效结果
                else:
                    # Handle cases where no valid match is found
                    print(f"未找到有效的判断结果。回复内容：{result}。正在重试...") 

            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)

    # Add the correctness results as a new column in the DataFrame
    qa_has_img_filtered['final_answer_correctness'] = final_answer_correctness
    qa_has_img_filtered['final_answer_correctness_judge_full'] = results

    return qa_has_img_filtered


def gpt4o_judge4qa_no_img_explanation1(qa_no_img):
    qa_no_img = qa_no_img.copy()
    client = openai.OpenAI()

    # System prompt tailored for checking answer correctness with flexible expression matching
    system_prompt = '''
    你是一个助教助手，负责判断学生答案的结论是否与标准答案的结论表达同样的意思。
    '''

    # Filter questions that are not proof questions
    # qa_no_img_filtered = qa_no_img[qa_no_img['is_proof_question'] == 'NO']
    qa_no_img_filtered = qa_no_img

    # Prepare to store GPT-4o ratings
    final_answer_correctness = []
    results = []

    for _, row in tqdm(qa_no_img_filtered.iterrows(), total=qa_no_img_filtered.shape[0]):
        # Prepare the content list for evaluation
        user_content = [
            {"type": "text", "text": "请根据以下题目信息和提供的标准答案判断学生答案是否正确:"},
            {"type": "text", "text": f"### 问题：\n{row['msg']}"},
            {"type": "text", "text": f"\n### 标准答案：\n{row['ans']}"},
            {"type": "text", "text": f"\n### 学生答案：\n{row['response']}"}
        ]

        requirement_prompt = '''
        请根据以上题目信息和标准答案，仅根据学生答案的最终结论或答案判断其是否正确，忽略过程的正确性。

        注意事项：
        1. 检查题目是否包含多个子问题：
        - 如果包含多个子问题，请逐一判断每个子问题的答案是否正确。只有当所有子问题的最终答案都正确时，整体答案才被视为正确。
        - 如果不包含子问题，则仅根据学生答案的最终结论或答案进行判断。
        2. 即使学生答案的表达方式与标准答案不同，只要最终结论或答案的意思相同，也应视为正确。可能的情况包括但不限于：
        - 学生答案和标准答案使用的语言不同，但意思相同。
        - 学生答案中的公式经过化简和变形后与标准答案的公式相同。
        - 学生答案采用了不同的表述形式，但语义相同。
        3. 请解释和分析学生答案与标准答案的最终结论或答案的异同之处。
        4. 如果由于学生答案不完整或缺失，或者学生答案没有按照题目要求给出结论，导致无法判断是否正确的，就判断为错误。
        5. 如果题目包含子问题，请为每个子问题提供“子问题X判断结果：正确”或“子问题X判断结果：错误”的形式。
        6. 最终判断结果应以“判断结果：正确”或“判断结果：错误”的形式给出。

        请按照以下格式回复：

        分析：
        [在此处填写详细的分析内容]

        [如果有子问题，则添加以下部分]
        子问题判断结果：
        子问题1判断结果：正确/错误
        子问题2判断结果：正确/错误
        ...
        子问题N判断结果：正确/错误

        判断结果：正确/错误
        '''
        user_content.append({'type': 'text', 'text': requirement_prompt})

        # Send the prompt to GPT-4o in a single message
        success = False
        while not success:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                )
                # Extract the model's response content
                result = response.choices[0].message.content.strip()
                # 无子问题时，只提取最终判断结果
                final_pattern = r'(?<!\d)判断结果：(正确|错误)'
                matches = re.findall(final_pattern, result)
                if matches:
                    # Select the last match from the list
                    final_result = matches[-1]
                    # Append the result to your existing lists
                    final_answer_correctness.append(final_result)
                    results.append(result)
                    # Indicate successful extraction
                    success = True  # 成功获取有效结果
                else:
                    # Handle cases where no valid match is found
                    print(f"未找到有效的判断结果。回复内容：{result}。正在重试...")
            except Exception as e:  # Catch any exception
                print(f"An error occurred: {e}")
                time.sleep(2)

    # Add the correctness results as a new column in the DataFrame
    qa_no_img_filtered['final_answer_correctness'] = final_answer_correctness
    qa_no_img_filtered['final_answer_correctness_judge_full'] = results

    return qa_no_img_filtered