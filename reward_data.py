import re
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')

MIN_ANSWER_COUNT = 2

def parse_body(body):
    """
    Returns the parsed body of a post, without HTML tags.

    Args:
        body: body of a post
    """
    # Remove code blocks
    reg_code_blocks = re.compile('(?s)<code>.*?</code>')
    body = reg_code_blocks.sub('', body)

    # Remove HTML tags
    reg_html_tags = re.compile('<.*?>')
    body = reg_html_tags.sub('', body)

    # Remove URLs
    reg_urls = re.compile('http\S+')
    body = reg_urls.sub('', body)

    # Tokenization
    words = word_tokenize(body)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the list of words back into a single string
    body = ' '.join(words)

    return body

def get_post_hierarchy(posts):
    """
    Returns a dictionary of posts with the following structure:
    {
        post_id: {
            'score': post_score,
            'body': post_body,
            'accepted_answer_id': post_accepted_answer_id,
            'answers': {
                answer_id: {
                    'score': answer_score,
                    'body': answer_body,
                },
                ...
            },
        },
        ...
    }

    Args:
        posts: list of posts
    """
    hierarchy = {}
    # Get parent posts
    for post in posts:
        if 'AcceptedAnswerId' not in post.attrib or 'AnswerCount' not in post.attrib or int(post.attrib['AnswerCount']) < MIN_ANSWER_COUNT:
            continue

        post_id = int(post.attrib['Id'])
        post_score = int(post.attrib['Score'])
        post_body = post.attrib['Body']
        post_accepted_answer_id = int(post.attrib['AcceptedAnswerId'])

        hierarchy[post_id] = {
            'score': post_score,
            'body': post_body,
            'accepted_answer_id': post_accepted_answer_id,
            'answers': {},
        }

    # Fill parent answers with children posts
    for post in posts:
        if 'ParentId' not in post.attrib or int(post.attrib['ParentId']) not in hierarchy:
            continue

        parent_id = int(post.attrib['ParentId'])
        post_id = int(post.attrib['Id'])
        post_score = int(post.attrib['Score'])
        post_body = post.attrib['Body']

        hierarchy[parent_id]['answers'][post_id] = {
            'score': post_score,
            'body': post_body,
        }

    return hierarchy

def preprocess_posts(posts):
    """
    Returns a dictionary of posts with the following structure:
    {
        post_id: {
            'score': post_score,
            'body': post_body,
            'answer_count': post_answer_count,
            'best_answer': {
                'score': best_answer_score,
                'body': best_answer_body,
            },
            'worst_answer': {
                'score': worst_answer_score,
                'body': worst_answer_body,
            },
        },
        ...
    }

    Args:
        posts: list of posts
    """
    hierarchy = get_post_hierarchy(posts)
    new_posts = {}

    for post_id, post in hierarchy.items():
        # Get best answer (accepted answer)
        if post['accepted_answer_id'] not in post['answers']:
            continue
        
        best_answer_score = post['answers'][post['accepted_answer_id']]['score']
        best_answer_body = post['answers'][post['accepted_answer_id']]['body']

        # Get worst answer (lowest score and not accepted answer)
        worst_answer_score = math.inf
        worst_answer_body = None
        for answer_id, answer in post['answers'].items():
            if answer_id == post['accepted_answer_id']:
                continue

            if answer['score'] < worst_answer_score:
                worst_answer_score = answer['score']
                worst_answer_body = answer['body']

        new_posts[post_id] = {
            'score': post['score'],
            'body': post['body'],
            'answer_count': len(post['answers']),
            'best_answer': {
                'score': best_answer_score,
                'body': best_answer_body,
            },
            'worst_answer': {
                'score': worst_answer_score,
                'body': worst_answer_body,
            },
        }

    return new_posts

def filter_posts(posts):
    """
    Filter posts that have too few score and that have too little score difference between the best and worst answer.
    """
    min_question_score, min_answer_count, min_answer_score_diff_perc = get_stats(posts)
    filtered_posts = {}

    for post_id, post in posts.items():
        if post['score'] < min_question_score or post['answer_count'] < min_answer_count or post['best_answer']['score'] - post['worst_answer']['score'] < min_answer_score_diff_perc * post['best_answer']['score']:
            continue

        filtered_posts[post_id] = post
    return filtered_posts

def posts_to_reward_data(preprocessed_posts):
    """
    Returns a list of tuples with the following structure:
    [
        {
            'label': label either 'positive' or 'negative',
            'chat': chat,
        },
        ...
    ]

    Args:
        preprocessed_posts: list of preprocessed posts
    """
    reward_data = []
    for _, post in preprocessed_posts.items():
        positive_chat = 'Human: ' + parse_body(post['body']) + '\nAssistant: ' + parse_body(post['best_answer']['body'])
        negative_chat = 'Human: ' + parse_body(post['body']) + '\nAssistant: ' + parse_body(post['worst_answer']['body'])

        reward_data.append({
            'label': 'positive',
            'chat': positive_chat,
        })
        reward_data.append({
            'label': 'negative',
            'chat': negative_chat,
        })

    return reward_data

def get_reward_data(posts):
    """
    Returns a list of tuples with the following structure:
    [
        {
            'label': label either 'positive' or 'negative',
            'chat': chat,
        },
        ...
    ]

    Args:
        posts: list of posts
    """
    posts = preprocess_posts(posts)
    print('\t📮 Number of posts:', len(posts))
    posts = filter_posts(posts)
    print('\t🧹 Number of filtered posts:', len(posts))
    reward_data = posts_to_reward_data(posts)
    return reward_data

def get_reward_data_from_and_save_to(file_path, save_file_path):
    """
    Saves the processed reward data to a file.
    
    Args:
        file_path: path to file
        save_file_path: path to save file
    """
    posts = ET.parse(file_path).getroot()
    reward_data = get_reward_data(posts)

    with open(save_file_path, 'w') as f:
        json.dump(reward_data, f, indent=4)

def show_stats(file_path):
    """
    Returns the following statistics:
    - median question score
    - median answer count
    - median accepted answer score
    - median worst answer score
    - median answer score difference
    - median answer score difference percentage
    """
    posts = ET.parse(file_path).getroot()
    posts = preprocess_posts(posts)

    question_scores = []
    answer_counts = []
    accepted_answer_scores = []
    worst_answer_scores = []
    answer_score_diffs = []
    answer_score_diff_percs = []
    for post in posts.values():
        question_score = int(post['score'])
        answer_count = int(post['answer_count'])
        accepted_answer_score = int(post['best_answer']['score'])
        worst_answer_score = int(post['worst_answer']['score'])
        answer_score_diff = accepted_answer_score - worst_answer_score
        answer_score_diff_perc = answer_score_diff / accepted_answer_score if accepted_answer_score > 0 else 0

        question_scores.append(question_score)
        answer_counts.append(answer_count)
        accepted_answer_scores.append(accepted_answer_score)
        worst_answer_scores.append(worst_answer_score)
        answer_score_diffs.append(answer_score_diff)
        answer_score_diff_percs.append(answer_score_diff_perc)

    # Figure with histograms
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].hist(question_scores, bins=100, log=True)
    axs[0, 0].set_title('Question score')
    axs[0, 1].hist(answer_counts, bins=100, log=True)
    axs[0, 1].set_title('Answer count')
    axs[0, 2].hist(accepted_answer_scores, bins=100, log=True)
    axs[0, 2].set_title('Accepted answer score')
    axs[1, 0].hist(worst_answer_scores, bins=100, log=True)
    axs[1, 0].set_title('Worst answer score')
    axs[1, 1].hist(answer_score_diffs, bins=100, log=True)
    axs[1, 1].set_title('Answer score difference')
    axs[1, 2].hist(answer_score_diff_percs, bins=100, log=True)
    axs[1, 2].set_title('Answer score difference percentage')

    # Show median values on figure
    median_question_score = np.median(question_scores)
    median_answer_count = np.median(answer_counts)
    median_accepted_answer_score = np.median(accepted_answer_scores)
    median_worst_answer_score = np.median(worst_answer_scores)
    median_answer_score_diff = np.median(answer_score_diffs)
    median_answer_score_diff_perc = np.median(answer_score_diff_percs)
    axs[0, 0].axvline(median_question_score, color='k', linestyle='dashed', linewidth=1)
    axs[0, 1].axvline(median_answer_count, color='k', linestyle='dashed', linewidth=1)
    axs[0, 2].axvline(median_accepted_answer_score, color='k', linestyle='dashed', linewidth=1)
    axs[1, 0].axvline(median_worst_answer_score, color='k', linestyle='dashed', linewidth=1)
    axs[1, 1].axvline(median_answer_score_diff, color='k', linestyle='dashed', linewidth=1)
    axs[1, 2].axvline(median_answer_score_diff_perc, color='k', linestyle='dashed', linewidth=1)
    
    plt.show()

def get_stats(posts):
    question_scores = []
    answer_counts = []
    accepted_answer_scores = []
    worst_answer_scores = []
    answer_score_diffs = []
    answer_score_diff_percs = []
    for post in posts.values():
        question_score = int(post['score'])
        answer_count = int(post['answer_count'])
        accepted_answer_score = int(post['best_answer']['score'])
        worst_answer_score = int(post['worst_answer']['score'])
        answer_score_diff = accepted_answer_score - worst_answer_score
        answer_score_diff_perc = answer_score_diff / accepted_answer_score if accepted_answer_score > 0 else 0

        question_scores.append(question_score)
        answer_counts.append(answer_count)
        accepted_answer_scores.append(accepted_answer_score)
        worst_answer_scores.append(worst_answer_score)
        answer_score_diffs.append(answer_score_diff)
        answer_score_diff_percs.append(answer_score_diff_perc)

    # min_question_score, which should be at least the top 10% of the question scores
    min_question_score = np.percentile(question_scores, 90)
    # min_answer_count, which should be at least the top 10% of the answer counts
    min_answer_count = np.percentile(answer_counts, 90)
    # min_answer_score_diff_perc, which should be at least the top 10% of the answer score difference percentages
    min_answer_score_diff_perc = np.percentile(answer_score_diff_percs, 90)

    return min_question_score, min_answer_count, min_answer_score_diff_perc

