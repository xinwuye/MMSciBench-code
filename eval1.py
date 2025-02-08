from data_util import *
from llm_apis.llm_as_util import *
import os
from concurrent.futures import ProcessPoolExecutor
# math
## multi_choice_has_img
def evaluate_multi_choice_has_img(model_name, read_path, output_path):
    print(f'evaluating math multi_choice_has_img {model_name}')
    
    # Ensure the output directory exists
    result_directory = os.path.dirname(output_path)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    
    # Load and process the data
    results = pd.read_csv(read_path)
    results['img'] = results['img'].apply(eval)

    # judged_results = gpt4o_judge4multi_choice(results)
    judged_results = results
    
    # Save the results to a CSV file
    judged_results.to_csv(output_path, index=False)
    print(f'{model_name} evaluation completed!')

# Define the parameters for each model
multi_choice_has_img_evaluations = [
    {
        'model_name': 'claude',
        'read_path': 'result_chn_nonre/math/multi_choice_has_img_claude.csv',
        'output_path': 'result_chn_nonre_reeval1/math/multi_choice_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_nonre/math/multi_choice_has_img_gpt4o.csv',
        'output_path': 'result_chn_nonre_reeval1/math/multi_choice_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_nonre/math/multi_choice_has_img_gemini.csv',
        'output_path': 'result_chn_nonre_reeval1/math/multi_choice_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_nonre/math/multi_choice_has_img_llama32.csv',
        'output_path': 'result_chn_nonre_reeval1/math/multi_choice_has_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_chn_stp_nonre/math/multi_choice_has_img_claude.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/multi_choice_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_stp_nonre/math/multi_choice_has_img_gpt4o.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/multi_choice_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_stp_nonre/math/multi_choice_has_img_gemini.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/multi_choice_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_stp_nonre/math/multi_choice_has_img_llama32.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/multi_choice_has_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_eng_stp_nonre/math/multi_choice_has_img_claude.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/multi_choice_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_eng_stp_nonre/math/multi_choice_has_img_gpt4o.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/multi_choice_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_eng_stp_nonre/math/multi_choice_has_img_gemini.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/multi_choice_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_eng_stp_nonre/math/multi_choice_has_img_llama32.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/multi_choice_has_img_llama32.csv',
    },
    # physics
    {
        'model_name': 'claude',
        'read_path': 'result_chn_nonre/physics/multi_choice_has_img_claude.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/multi_choice_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_nonre/physics/multi_choice_has_img_gpt4o.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/multi_choice_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_nonre/physics/multi_choice_has_img_gemini.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/multi_choice_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_nonre/physics/multi_choice_has_img_llama32.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/multi_choice_has_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_chn_stp_nonre/physics/multi_choice_has_img_claude.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/multi_choice_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_stp_nonre/physics/multi_choice_has_img_gpt4o.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/multi_choice_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_stp_nonre/physics/multi_choice_has_img_gemini.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/multi_choice_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_stp_nonre/physics/multi_choice_has_img_llama32.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/multi_choice_has_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_eng_stp_nonre/physics/multi_choice_has_img_claude.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/multi_choice_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_eng_stp_nonre/physics/multi_choice_has_img_gpt4o.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/multi_choice_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_eng_stp_nonre/physics/multi_choice_has_img_gemini.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/multi_choice_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_eng_stp_nonre/physics/multi_choice_has_img_llama32.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/multi_choice_has_img_llama32.csv',
    },
]

## multi_choice_no_img
def evaluate_multi_choice_no_img(model_name, read_path, output_path):
    print(f'evaluating math multi_choice_no_img {model_name}')
    
    # Ensure the output directory exists
    result_directory = os.path.dirname(output_path)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    
    # Load and process the data
    results = pd.read_csv(read_path)

    # judged_results = gpt4o_judge4multi_choice(results)
    judged_results = results
    
    # Save the results to a CSV file
    judged_results.to_csv(output_path, index=False)
    print(f'{model_name} evaluation completed!')

# Define the parameters for each model
multi_choice_no_img_evaluations = [
    {
        'model_name': 'claude',
        'read_path': 'result_chn_nonre/math/multi_choice_no_img_claude.csv',
        'output_path': 'result_chn_nonre_reeval1/math/multi_choice_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_nonre/math/multi_choice_no_img_gpt4o.csv',
        'output_path': 'result_chn_nonre_reeval1/math/multi_choice_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_nonre/math/multi_choice_no_img_gemini.csv',
        'output_path': 'result_chn_nonre_reeval1/math/multi_choice_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_nonre/math/multi_choice_no_img_llama32.csv',
        'output_path': 'result_chn_nonre_reeval1/math/multi_choice_no_img_llama32.csv',
    },
    {
        'model_name': 'qwen25math72b',
        'read_path': 'result_chn_nonre/math/multi_choice_no_img_qwen25math72b.csv',
        'output_path': 'result_chn_nonre_reeval1/math/multi_choice_no_img_qwen25math72b.csv',
    },
    {
        'model_name': 'deepseekmath7b',
        'read_path': 'result_chn_nonre/math/multi_choice_no_img_deepseekmath7b.csv',
        'output_path': 'result_chn_nonre_reeval1/math/multi_choice_no_img_deepseekmath7b.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_chn_stp_nonre/math/multi_choice_no_img_claude.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/multi_choice_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_stp_nonre/math/multi_choice_no_img_gpt4o.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/multi_choice_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_stp_nonre/math/multi_choice_no_img_gemini.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/multi_choice_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_stp_nonre/math/multi_choice_no_img_llama32.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/multi_choice_no_img_llama32.csv',
    },
    {
        'model_name': 'qwen25math72b',
        'read_path': 'result_chn_stp_nonre/math/multi_choice_no_img_qwen25math72b.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/multi_choice_no_img_qwen25math72b.csv',
    },
    {
        'model_name': 'deepseekmath7b',
        'read_path': 'result_chn_stp_nonre/math/multi_choice_no_img_deepseekmath7b.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/multi_choice_no_img_deepseekmath7b.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_eng_stp_nonre/math/multi_choice_no_img_claude.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/multi_choice_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_eng_stp_nonre/math/multi_choice_no_img_gpt4o.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/multi_choice_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_eng_stp_nonre/math/multi_choice_no_img_gemini.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/multi_choice_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_eng_stp_nonre/math/multi_choice_no_img_llama32.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/multi_choice_no_img_llama32.csv',
    },
    {
        'model_name': 'qwen25math72b',
        'read_path': 'result_eng_stp_nonre/math/multi_choice_no_img_qwen25math72b.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/multi_choice_no_img_qwen25math72b.csv',
    },
    {
        'model_name': 'deepseekmath7b',
        'read_path': 'result_eng_stp_nonre/math/multi_choice_no_img_deepseekmath7b.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/multi_choice_no_img_deepseekmath7b.csv',
    },
    # physics
    {
        'model_name': 'claude',
        'read_path': 'result_chn_nonre/physics/multi_choice_no_img_claude.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/multi_choice_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_nonre/physics/multi_choice_no_img_gpt4o.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/multi_choice_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_nonre/physics/multi_choice_no_img_gemini.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/multi_choice_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_nonre/physics/multi_choice_no_img_llama32.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/multi_choice_no_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_chn_stp_nonre/physics/multi_choice_no_img_claude.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/multi_choice_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_stp_nonre/physics/multi_choice_no_img_gpt4o.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/multi_choice_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_stp_nonre/physics/multi_choice_no_img_gemini.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/multi_choice_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_stp_nonre/physics/multi_choice_no_img_llama32.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/multi_choice_no_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_eng_stp_nonre/physics/multi_choice_no_img_claude.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/multi_choice_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_eng_stp_nonre/physics/multi_choice_no_img_gpt4o.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/multi_choice_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_eng_stp_nonre/physics/multi_choice_no_img_gemini.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/multi_choice_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_eng_stp_nonre/physics/multi_choice_no_img_llama32.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/multi_choice_no_img_llama32.csv',
    },
]

## qa_has_img
def evaluate_qa_has_img(model_name, read_path, output_path):
    print(f'evaluating math qa_has_img {model_name}')
    
    # Ensure the output directory exists
    result_directory = os.path.dirname(output_path)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    
    # Load and filter the data
    results = pd.read_csv(read_path)
    results['content_img'] = results['content_img'].apply(eval)
    results['answer_img'] = results['answer_img'].apply(eval)
    judged_results = gpt4o_judge4qa_has_img_explanation1(results)
    
    # Save the results to a CSV file
    judged_results.to_csv(output_path, index=False)
    print(f'{model_name} evaluation completed!')

# Define the parameters for each model
qa_has_img_evaluations = [
    {
        'model_name': 'claude',
        'read_path': 'result_chn_nonre/math/qa_has_img_claude.csv',
        'output_path': 'result_chn_nonre_reeval1/math/qa_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_nonre/math/qa_has_img_gpt4o.csv',
        'output_path': 'result_chn_nonre_reeval1/math/qa_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_nonre/math/qa_has_img_gemini.csv',
        'output_path': 'result_chn_nonre_reeval1/math/qa_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_nonre/math/qa_has_img_llama32.csv',
        'output_path': 'result_chn_nonre_reeval1/math/qa_has_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_chn_stp_nonre/math/qa_has_img_claude.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/qa_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_stp_nonre/math/qa_has_img_gpt4o.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/qa_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_stp_nonre/math/qa_has_img_gemini.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/qa_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_stp_nonre/math/qa_has_img_llama32.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/qa_has_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_eng_stp_nonre/math/qa_has_img_claude.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/qa_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_eng_stp_nonre/math/qa_has_img_gpt4o.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/qa_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_eng_stp_nonre/math/qa_has_img_gemini.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/qa_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_eng_stp_nonre/math/qa_has_img_llama32.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/qa_has_img_llama32.csv',
    },
    # physics
    {
        'model_name': 'claude',
        'read_path': 'result_chn_nonre/physics/qa_has_img_claude.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/qa_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_nonre/physics/qa_has_img_gpt4o.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/qa_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_nonre/physics/qa_has_img_gemini.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/qa_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_nonre/physics/qa_has_img_llama32.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/qa_has_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_chn_stp_nonre/physics/qa_has_img_claude.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/qa_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_stp_nonre/physics/qa_has_img_gpt4o.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/qa_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_stp_nonre/physics/qa_has_img_gemini.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/qa_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_stp_nonre/physics/qa_has_img_llama32.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/qa_has_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_eng_stp_nonre/physics/qa_has_img_claude.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/qa_has_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_eng_stp_nonre/physics/qa_has_img_gpt4o.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/qa_has_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_eng_stp_nonre/physics/qa_has_img_gemini.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/qa_has_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_eng_stp_nonre/physics/qa_has_img_llama32.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/qa_has_img_llama32.csv',
    },
]


## qa_no_img
def evaluate_qa_no_img(model_name, read_path, output_path):
    print(f'evaluating math qa_no_img {model_name}')
    
    # Ensure the output directory exists
    result_directory = os.path.dirname(output_path)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    
    # Load and filter the data
    results = pd.read_csv(read_path)
    judged_results = gpt4o_judge4qa_no_img_explanation1(results)
    
    # Save the results to a CSV file
    judged_results.to_csv(output_path, index=False)
    print(f'{model_name} evaluation completed!')

# Define the parameters for each model
qa_no_img_evaluations = [
    {
        'model_name': 'claude',
        'read_path': 'result_chn_nonre/math/qa_no_img_claude.csv',
        'output_path': 'result_chn_nonre_reeval1/math/qa_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_nonre/math/qa_no_img_gpt4o.csv',
        'output_path': 'result_chn_nonre_reeval1/math/qa_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_nonre/math/qa_no_img_gemini.csv',
        'output_path': 'result_chn_nonre_reeval1/math/qa_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_nonre/math/qa_no_img_llama32.csv',
        'output_path': 'result_chn_nonre_reeval1/math/qa_no_img_llama32.csv',
    },
    {
        'model_name': 'qwen25math72b',
        'read_path': 'result_chn_nonre/math/qa_no_img_qwen25math72b.csv',
        'output_path': 'result_chn_nonre_reeval1/math/qa_no_img_qwen25math72b.csv',
    },
    {
        'model_name': 'deepseekmath7b',
        'read_path': 'result_chn_nonre/math/qa_no_img_deepseekmath7b.csv',
        'output_path': 'result_chn_nonre_reeval1/math/qa_no_img_deepseekmath7b.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_chn_stp_nonre/math/qa_no_img_claude.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/qa_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_stp_nonre/math/qa_no_img_gpt4o.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/qa_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_stp_nonre/math/qa_no_img_gemini.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/qa_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_stp_nonre/math/qa_no_img_llama32.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/qa_no_img_llama32.csv',
    },
    {
        'model_name': 'qwen25math72b',
        'read_path': 'result_chn_stp_nonre/math/qa_no_img_qwen25math72b.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/qa_no_img_qwen25math72b.csv',
    },
    {
        'model_name': 'deepseekmath7b',
        'read_path': 'result_chn_stp_nonre/math/qa_no_img_deepseekmath7b.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/math/qa_no_img_deepseekmath7b.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_eng_stp_nonre/math/qa_no_img_claude.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/qa_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_eng_stp_nonre/math/qa_no_img_gpt4o.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/qa_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_eng_stp_nonre/math/qa_no_img_gemini.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/qa_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_eng_stp_nonre/math/qa_no_img_llama32.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/qa_no_img_llama32.csv',
    },
    {
        'model_name': 'qwen25math72b',
        'read_path': 'result_eng_stp_nonre/math/qa_no_img_qwen25math72b.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/qa_no_img_qwen25math72b.csv',
    },
    {
        'model_name': 'deepseekmath7b',
        'read_path': 'result_eng_stp_nonre/math/qa_no_img_deepseekmath7b.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/math/qa_no_img_deepseekmath7b.csv',
    },
    # physics
    {
        'model_name': 'claude',
        'read_path': 'result_chn_nonre/physics/qa_no_img_claude.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/qa_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_nonre/physics/qa_no_img_gpt4o.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/qa_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_nonre/physics/qa_no_img_gemini.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/qa_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_nonre/physics/qa_no_img_llama32.csv',
        'output_path': 'result_chn_nonre_reeval1/physics/qa_no_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_chn_stp_nonre/physics/qa_no_img_claude.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/qa_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_chn_stp_nonre/physics/qa_no_img_gpt4o.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/qa_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_chn_stp_nonre/physics/qa_no_img_gemini.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/qa_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_chn_stp_nonre/physics/qa_no_img_llama32.csv',
        'output_path': 'result_chn_stp_nonre_reeval1/physics/qa_no_img_llama32.csv',
    },

    {
        'model_name': 'claude',
        'read_path': 'result_eng_stp_nonre/physics/qa_no_img_claude.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/qa_no_img_claude.csv',
    },
    {
        'model_name': 'gpt-4o',
        'read_path': 'result_eng_stp_nonre/physics/qa_no_img_gpt4o.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/qa_no_img_gpt4o.csv',
    },
    {
        'model_name': 'gemini',
        'read_path': 'result_eng_stp_nonre/physics/qa_no_img_gemini.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/qa_no_img_gemini.csv',
    },
    {
        'model_name': 'llama 3.2',
        'read_path': 'result_eng_stp_nonre/physics/qa_no_img_llama32.csv',
        'output_path': 'result_eng_stp_nonre_reeval1/physics/qa_no_img_llama32.csv',
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
        )
        for eval in multi_choice_has_img_evaluations
    ])
    futures.extend([
        executor.submit(
            evaluate_multi_choice_no_img,
            eval['model_name'],
            eval['read_path'],
            eval['output_path'],
        )
        for eval in multi_choice_no_img_evaluations
    ])
    futures.extend([
        executor.submit(
            evaluate_qa_has_img,
            eval['model_name'],
            eval['read_path'],
            eval['output_path'],
        )
        for eval in qa_has_img_evaluations
    ])
    futures.extend([
        executor.submit(
            evaluate_qa_no_img,
            eval['model_name'],
            eval['read_path'],
            eval['output_path'],
        )
        for eval in qa_no_img_evaluations
    ])
    
    # Wait for all threads to complete
    for future in futures:
        future.result()