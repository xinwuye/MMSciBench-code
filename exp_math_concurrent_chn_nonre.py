from data_util import *
from llm_apis.llm_as_util import *
from llm_apis import claude_chn_nonre as la_claude
from llm_apis import fireworksai_chn_nonre as la_fireworksai
from llm_apis import gemini_chn_nonre as la_gemini
from llm_apis import gpt_chn_nonre as la_gpt
import os
from concurrent.futures import ProcessPoolExecutor
# math
## multi_choice_has_img
def evaluate_multi_choice_has_img(model_name, read_path, output_path, evaluation_function):
    print(f'evaluating math multi_choice_has_img {model_name}')
    
    # Ensure the output directory exists
    result_directory = os.path.dirname(output_path)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    
    # Load and process the data
    multi_choice_has_img = read_processed_data_has_img(read_path)
    
    # Run the evaluation function for the specific model
    results = evaluation_function(multi_choice_has_img)

    judged_results = gpt4o_judge4multi_choice(results)
    
    # Save the results to a CSV file
    judged_results.to_csv(output_path, index=False)
    print(f'{model_name} evaluation completed!')

# Define the parameters for each model
multi_choice_has_img_evaluations = [
    {
        'model_name': 'claude',
        'read_path': 'data/math/multi_choice_has_img_with_categories.csv',
        'output_path': 'result_chn_nonre/math/multi_choice_has_img_claude.csv',
        'evaluation_function': la_claude.claude_multi_choice_has_img
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'data/math/multi_choice_has_img_with_categories.csv',
        'output_path': 'result_chn_nonre/math/multi_choice_has_img_gpt4o.csv',
        'evaluation_function': la_gpt.gpt4o_multi_choice_has_img
    },
    {
        'model_name': 'gemini',
        'read_path': 'data/math/multi_choice_has_img_with_categories.csv',
        'output_path': 'result_chn_nonre/math/multi_choice_has_img_gemini.csv',
        'evaluation_function': la_gemini.gemini_multi_choice_has_img
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'data/math/multi_choice_has_img_with_categories.csv',
        'output_path': 'result_chn_nonre/math/multi_choice_has_img_llama32.csv',
        'evaluation_function': la_fireworksai.llama32_multi_choice_has_img
    },
]

## multi_choice_no_img
def evaluate_multi_choice_no_img(model_name, read_path, output_path, evaluation_function):
    print(f'evaluating math multi_choice_no_img {model_name}')
    
    # Ensure the output directory exists
    result_directory = os.path.dirname(output_path)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    
    # Load and process the data
    multi_choice_no_img = read_processed_data_no_img(read_path)
    
    # Run the evaluation function for the specific model
    results = evaluation_function(multi_choice_no_img)

    judged_results = gpt4o_judge4multi_choice(results)
    
    # Save the results to a CSV file
    judged_results.to_csv(output_path, index=False)
    print(f'{model_name} evaluation completed!')

# Define the parameters for each model
multi_choice_no_img_evaluations = [
    {
        'model_name': 'claude',
        'read_path': 'data/math/multi_choice_no_img_with_categories.csv',
        'output_path': 'result_chn_nonre/math/multi_choice_no_img_claude.csv',
        'evaluation_function': la_claude.claude_multi_choice_no_img
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'data/math/multi_choice_no_img_with_categories.csv',
        'output_path': 'result_chn_nonre/math/multi_choice_no_img_gpt4o.csv',
        'evaluation_function': la_gpt.gpt4o_multi_choice_no_img
    },
    {
        'model_name': 'gemini',
        'read_path': 'data/math/multi_choice_no_img_with_categories.csv',
        'output_path': 'result_chn_nonre/math/multi_choice_no_img_gemini.csv',
        'evaluation_function': la_gemini.gemini_multi_choice_no_img
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'data/math/multi_choice_no_img_with_categories.csv',
        'output_path': 'result_chn_nonre/math/multi_choice_no_img_llama32.csv',
        'evaluation_function': la_fireworksai.llama32_multi_choice_no_img
    },
]

## qa_has_img
def evaluate_qa_has_img(model_name, read_path, output_path, evaluation_function):
    print(f'evaluating math qa_has_img {model_name}')
    
    # Ensure the output directory exists
    result_directory = os.path.dirname(output_path)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    
    # Load and filter the data
    qa_has_img = read_processed_data_has_img(read_path)
    qa_has_img = qa_has_img[qa_has_img['is_proof_question'] == 'NO']
    
    # Run the evaluation function for the specific model
    results = evaluation_function(qa_has_img)

    # Save the results to a CSV file
    results.to_csv(output_path, index=False)
    print(f'{model_name} evaluation completed!')

# Define the parameters for each model
qa_has_img_evaluations = [
    {
        'model_name': 'claude',
        'read_path': 'data/math/QA_has_img_with_proof_tags_and_categories.csv',
        'output_path': 'result_chn_nonre/math/qa_has_img_claude.csv',
        'evaluation_function': la_claude.claude_qa_has_img
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'data/math/QA_has_img_with_proof_tags_and_categories.csv',
        'output_path': 'result_chn_nonre/math/qa_has_img_gpt4o.csv',
        'evaluation_function': la_gpt.gpt4o_qa_has_img
    },
    {
        'model_name': 'gemini',
        'read_path': 'data/math/QA_has_img_with_proof_tags_and_categories.csv',
        'output_path': 'result_chn_nonre/math/qa_has_img_gemini.csv',
        'evaluation_function': la_gemini.gemini_qa_has_img
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'data/math/QA_has_img_with_proof_tags_and_categories.csv',
        'output_path': 'result_chn_nonre/math/qa_has_img_llama32.csv',
        'evaluation_function': la_fireworksai.llama32_qa_has_img
    },
]


## qa_no_img
def evaluate_qa_no_img(model_name, read_path, output_path, evaluation_function):
    print(f'evaluating math qa_no_img {model_name}')
    
    # Ensure the output directory exists
    result_directory = os.path.dirname(output_path)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    
    # Load and filter the data
    qa_no_img = read_processed_data_no_img(read_path)
    qa_no_img = qa_no_img[qa_no_img['is_proof_question'] == 'NO']
    
    # Run the evaluation function for the specific model
    results = evaluation_function(qa_no_img)
    
    # Save the results to a CSV file
    results.to_csv(output_path, index=False)
    print(f'{model_name} evaluation completed!')

# Define the parameters for each model
qa_no_img_evaluations = [
    {
        'model_name': 'claude',
        'read_path': 'data/math/QA_no_img_with_proof_tags_and_categories.csv',
        'output_path': 'result_chn_nonre/math/qa_no_img_claude.csv',
        'evaluation_function': la_claude.claude_qa_no_img
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'data/math/QA_no_img_with_proof_tags_and_categories.csv',
        'output_path': 'result_chn_nonre/math/qa_no_img_gpt4o.csv',
        'evaluation_function': la_gpt.gpt4o_qa_no_img
    },
    {
        'model_name': 'gemini',
        'read_path': 'data/math/QA_no_img_with_proof_tags_and_categories.csv',
        'output_path': 'result_chn_nonre/math/qa_no_img_gemini.csv',
        'evaluation_function': la_gemini.gemini_qa_no_img
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'data/math/QA_no_img_with_proof_tags_and_categories.csv',
        'output_path': 'result_chn_nonre/math/qa_no_img_llama32.csv',
        'evaluation_function': la_fireworksai.llama32_qa_no_img
    },
]

# Use ThreadPoolExecutor for concurrent execution
with ProcessPoolExecutor(max_workers=16) as executor:
    futures = []
    futures.extend([
        executor.submit(
            evaluate_multi_choice_has_img,
            eval['model_name'],
            eval['read_path'],
            eval['output_path'],
            eval['evaluation_function']
        )
        for eval in multi_choice_has_img_evaluations
    ])
    futures.extend([
        executor.submit(
            evaluate_multi_choice_no_img,
            eval['model_name'],
            eval['read_path'],
            eval['output_path'],
            eval['evaluation_function']
        )
        for eval in multi_choice_no_img_evaluations
    ])
    futures.extend([
        executor.submit(
            evaluate_qa_has_img,
            eval['model_name'],
            eval['read_path'],
            eval['output_path'],
            eval['evaluation_function']
        )
        for eval in qa_has_img_evaluations
    ])
    futures.extend([
        executor.submit(
            evaluate_qa_no_img,
            eval['model_name'],
            eval['read_path'],
            eval['output_path'],
            eval['evaluation_function']
        )
        for eval in qa_no_img_evaluations
    ])
    
    # Wait for all threads to complete
    for future in futures:
        future.result()