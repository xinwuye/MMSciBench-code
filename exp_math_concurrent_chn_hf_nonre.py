from data_util import *
from llm_apis.llm_as_util import *
from llm_apis import hf_chn_nonre as la_hf
import os
from concurrent.futures import ProcessPoolExecutor
# math

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
        'model_name': 'qwen25math72b',
        'read_path': 'data/math/multi_choice_no_img_with_categories.csv',
        'output_path': 'result_chn_nonre/math/multi_choice_no_img_qwen25math72b.csv',
        'evaluation_function': la_hf.qwen25math72b_multi_choice_no_img
    },
    {
        'model_name': 'deepseekmath7b',
        'read_path': 'data/math/multi_choice_no_img_with_categories.csv',
        'output_path': 'result_chn_nonre/math/multi_choice_no_img_deepseekmath7b.csv',
        'evaluation_function': la_hf.deepseekmath7b_multi_choice_no_img
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
        'model_name': 'qwen25math72b',
        'read_path': 'data/math/QA_no_img_with_proof_tags_and_categories.csv',
        'output_path': 'result_chn_nonre_reeval1_correct/math/qa_no_img_qwen25math72b.csv',
        'evaluation_function': la_hf.qwen25math72b_qa_no_img
    },
    {
        'model_name': 'deepseekmath7b',
        'read_path': 'data/math/QA_no_img_with_proof_tags_and_categories.csv',
        'output_path': 'result_chn_nonre_reeval1_correct/math/qa_no_img_deepseekmath7b.csv',
        'evaluation_function': la_hf.deepseekmath7b_qa_no_img
    },
]

# Use ThreadPoolExecutor for concurrent execution
with ProcessPoolExecutor(max_workers=16) as executor:
    futures = []
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