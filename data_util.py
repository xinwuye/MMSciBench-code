import pandas as pd
from bs4 import BeautifulSoup
import re
import json
import os
from openai import OpenAI
import openai
import time
from tqdm import tqdm
import math
from urllib.parse import unquote


def read_processed_data_has_img(path):
    processed_data_has_img = pd.read_csv(path)
    if 'content_img' in processed_data_has_img.columns:
        # the col content_img and answer_img are lists but they are stored as strings
        processed_data_has_img['content_img'] = processed_data_has_img['content_img'].apply(eval)
        processed_data_has_img['answer_img'] = processed_data_has_img['answer_img'].apply(eval)
    elif 'img' in processed_data_has_img.columns:
        processed_data_has_img['img'] = processed_data_has_img['img'].apply(eval)

    return processed_data_has_img


def read_processed_data_no_img(path):
    processed_data_no_img = pd.read_csv(path)

    return processed_data_no_img