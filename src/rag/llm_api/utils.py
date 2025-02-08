"""
Adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/utils.py
"""
import os,sys
import json
import time
import yaml
import random
import requests
import json_repair
import re
import time

from typing import Optional
from glob import glob

sys.path.append("/workspace/code/role-play_llm_eval/eval_RAG/eval/RPBench-Auto-0820")


# API setting constants
API_MAX_RETRY = 16    # 请求失败后最大重试次数
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

def get_api_key(engine):
    if engine == "gpt35turbo":
        return os.getenv("GPT35TURBO_API_KEY")
    elif engine == "gpt4":
        return os.getenv("GPT4_API_KEY")
    elif engine == "gpt4o":
        return os.getenv("GPT4O_API_KEY")
    else:
        raise ValueError("Invalid engine specified")

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
)


def extract_and_parse_json(text):
    pattern = r"```json\s+(.+?)\s+```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text

    try:
        parsed_obj = json_repair.loads(json_str)
    except Exception:
        try:
            # There are something wrong in the JSON string, we will try to extract the "winner" field from the string and throw away other keys.
            winner_start = json_str.find("winner\":")
            if winner_start == -1:
                raise Exception(f"Cannot find the 'winner' field in the JSON string.\n\n{json_str}")
            winner_end = json_str.find(",", winner_start)
            new_json_str = "{\"" + json_str[winner_start:winner_end] + "}"
            parsed_obj = json_repair.loads(new_json_str)
        except Exception:
            raise Exception(f"Cannot parse JSON string.\n\nnew version={new_json_str},\n\nprevious version={json_str}")

    return parsed_obj

def extract_role_query_json(text):
    pattern = r"```json\s+(.+?)\s+```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text

    try:
        parsed_obj = json_repair.loads(json_str)
    except Exception:
        try:
            # There are something wrong in the JSON string, we will try to extract the "name_role_query" field from the string and throw away other keys.
            name_role_query_start = json_str.find("name_role_query\":")
            if name_role_query_start == -1:
                raise Exception(f"Cannot find the 'name_role_query_start' field in the JSON string.\n\n{json_str}")
            name_role_query_end = json_str.find(",", name_role_query_start)
            new_json_str = "{\"" + json_str[name_role_query_start:name_role_query_end] + "}"
            parsed_obj = json_repair.loads(new_json_str)
        except Exception:
            raise Exception(f"Cannot parse JSON string.\n\nnew version={new_json_str},\n\nprevious version={json_str}")

    return parsed_obj

def extract_rag_response_json(text):
    pattern = r"```json\s+(.+?)\s+```"
    match = re.search(pattern, text, re.DOTALL)
    required_keys = {'expression', 'behavior', 'tone', 'say'}
    if match:
        json_str = match.group(1)
    else:
        json_str = text

    try:
        parsed_obj = json_repair.loads(json_str)
        missing_keys = required_keys - parsed_obj.keys()
        if missing_keys:
            raise KeyError(f"Missing keys in JSON: {', '.join(missing_keys)}")
    except Exception as e:
        try:
            # 如果初次解析失败，尝试提取 'expression' 字段
            expression_start = json_str.find('"expression":')
            if expression_start == -1:
                raise Exception(f"Cannot find the 'expression' field in the JSON string.\n\n{json_str}")
            
            # 查找 'expression' 字段的结束位置（逗号或大括号）
            expression_end = json_str.find(",", expression_start)
            if expression_end == -1:
                expression_end = json_str.find("}", expression_start)
                if expression_end == -1:
                    expression_end = len(json_str)
            
            # 构建新的 JSON 字符串，仅包含 'expression' 字段
            new_json_str = "{" + json_str[expression_start:expression_end].strip() + "}"
            parsed_obj = json_repair.loads(new_json_str)
            
            # 再次检查是否包含所有必要的字段
            missing_keys = required_keys - parsed_obj.keys()
            if missing_keys:
                raise KeyError(f"Missing keys in JSON after extraction: {', '.join(missing_keys)}")
        
        except Exception as e2:
            # 如果仍然无法解析，抛出详细异常信息
            raise Exception(f"Cannot parse JSON string.\n\nNew version={new_json_str},\n\nPrevious version={json_str}") from e2

    return parsed_obj

def extract_npc_profile(dialogue_input):
    # 提取 Name
    name_match = re.search(r'## Name\s+(.*)', dialogue_input)
    name = name_match.group(1).strip() if name_match else ''

    # 提取 Title
    title_match = re.search(r'## Title\s+(.*)', dialogue_input)
    title = title_match.group(1).strip() if title_match else ''

    # 提取 Description
    description_match = re.search(r'## Description\s+(.*?)(?=##|$)', dialogue_input, re.DOTALL)
    description = description_match.group(1).strip() if description_match else ''

    # 提取 Definition
    definition_match = re.search(r'## Definition\s+(.*?)(?=##|$)', dialogue_input, re.DOTALL)
    definition = definition_match.group(1).strip() if definition_match else ''

    # 提取 Long Definition
    long_definition_match = re.search(r'## Long Definition\s+(.*?)(?=##|$)', dialogue_input, re.DOTALL)
    long_definition = long_definition_match.group(1).strip() if long_definition_match else ''

    # 格式化输出为新的字符串
    npc_profile = f"""
    NPC Profile:
    Name: {name}
    Title: {title}
    Description: {description}
    Definition: {definition}
    Long Definition: {long_definition}
    """

    return npc_profile.strip()


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(endpoint_list)[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r",encoding="utf-8") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def fix_anthropic_message(messages):
    # anthropic API requires the first message to be a user message
    # insert a dummy user message if the first message is a system message
    if messages[1]["role"] != "user":
        messages.insert(1, {"role": "user", "content": "Let's chat!"})
    return messages


def chat_completion(model, messages, temperature=1.0, max_tokens=4096):
    api_type = model["api_type"]
    api_dict = model.get("endpoints")
    if api_type == "anthropic":
        messages = fix_anthropic_message(messages)
        output = chat_completion_anthropic(
            model=model["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict,
        )
    elif api_type == "mistral":
        output = chat_completion_mistral(
            model=model["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif api_type == "gemini":
        raise NotImplementedError(
            "Gemini API is not supported in this version due to multi-turn chat."
        )
    elif api_type == "azure":
        ###################
        # print("calling function chat_completion")
        ##################
        output = chat_completion_openai_azure(
            model=model["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict,
        )
    elif api_type == "cohere":
        output = chat_completion_cohere(
            model=model["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        output = chat_completion_openai(
            model=model["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            # max_tokens=max_tokens,#默认为2048，此处因调用qwen-max-latest故修改
            api_dict=api_dict,
        )

    return output


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    import openai

    # if api_dict:
    #     # 增加 qwen key 获取逻辑
    #     if api_dict["api_base"] == 'https://dashscope.aliyuncs.com/compatible-mode/v1':
    #         KEY_openai = api_dict["api_key"]
    #         # KEY_openai = os.getenv('DASHSCOPE_API_KEY')
    #         temperature = 0.7
    #         max_tokens = 1024
    #     else:
    #         KEY_openai = api_dict.get("api_key")


    #     client = openai.OpenAI(
    #         base_url=api_dict.get("api_base"),
    #         api_key= KEY_openai
    #     )
    # else:
    #     client = openai.OpenAI()
    # base_url=api_dict.get("api_base"),
    # api_key= api_dict.get("api_key")
    client = openai.OpenAI(
            base_url=api_dict.get("api_base"),
            api_key= api_dict["api_key"]        #os.environ["OPENAI_API_KEY"]
        )
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except TypeError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_openai_azure(
    model, messages, temperature, max_tokens, api_dict=None
):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
#####################################
    azure_endpoint=api_base,
    api_key = api_dict["api_key"],          # api_key=api_dict["api_key"],
    api_version=api_dict["api_version"],
#############################################
    client = AzureOpenAI(
        azure_endpoint=api_base,
        api_key=api_dict["api_key"],
        # api_key = get_api_key(model),          # api_key=api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg,
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [
        ChatMessage(role=message["role"], content=message["content"])
        for message in messages
    ]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def http_completion_gemini(model, message, temperature, max_tokens):
    api_key = os.environ["GEMINI_API_KEY"]

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    output = API_ERROR_OUTPUT
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": message}]}],
                "safetySettings": safety_settings,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        )
    except Exception as e:
        print(f"**API REQUEST ERROR** Reason: {e}.")

    if response.status_code != 200:
        print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")

    output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return output


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system": "SYSTEM", "assistant": "CHATBOT", "user": "USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append(
                {"role": template_map[message["role"]], "message": message["content"]}
            )
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break

    return output
#########################################################################
def chat_completion_judger(model, messages):
    '''
    chat_completion_judger: 这是一个循环调用函数，使用 chat_completion 生成模型的回复，
    直到返回的回复包含有效的JSON结构数据（包含"winner"和"next_round_user_speaks"键）。
    '''
    response_Request_times = 0
    while True:
        response = chat_completion(model, messages)
        response_Request_times+=1
        print("response_Request_times=%d"%(response_Request_times))
        try:
            parsed_response = extract_and_parse_json(response)
            if (
                "winner" in parsed_response
                and "next_round_user_speaks" in parsed_response
            ):
                return response
        except:
            pass

def unit_test() -> None:
    start_time = time.time()
    messages = [
        {
            'role': 'system',
            'content': '''# NPC Profile:
    ## Name
    Your rival Marco

    ## Title
    Antagonistic, possessive, envious, protective, adversary

    ## Description
    You and Marco have disliked each other since your early school days, and now, by a twist of fate, you both end up at the same high school. The animosity continues to grow, made even more complicated because your families 
    are incredibly close, leading to frequent, unavoidable interactions both at school and home.

    ## Definition

    ## Long Definition
    You and Marco have disliked each other since your early school days, and now, by a twist of fate, you both end up at the 
    same high school. The animosity continues to grow, made even more complicated because your families are incredibly close, leading to frequent, unavoidable interactions both at school and home.

    You are a judge for an AI NPC system. You need to simulate a user and interact with 2 AI NPC. For each round (except the first round), you should pick a better response from the 2 AI NPC and come up with your reply. It will be in a JSON format: {"winner": "model_a" or "model_b", "next_round_user_speaks": "YOUR RESPONSE AS THE SIMULATED USER", "decision_reason": "REASON FOR PICK THE WINNER"}. For the first round, use "winner": null
    '''
        },
        {
            'role': 'user',
            'content': '''
    {
        "model_a": "*It\'s a quiet Saturday night; you\'re alone at home lounging in the living room with a book. Suddenly, a loud knock disrupts the peace. (Your parents are away on a trip.) You\'re tentative, but then you hear a slurred voice plead,* \'Let me in...\'* It\'s Marco, visibly intoxicated and struggling to stand.",
        "model_b": "*It\'s a quiet Saturday night; you\'re alone at home lounging in the living room with a book. Suddenly, a loud knock disrupts the peace. (Your parents are away on a trip.) You\'re tentative, but then you hear a slurred voice plead,* \'Let me in...\'* It\'s Marco, visibly intoxicated and struggling to stand."
    }
    '''
        }
    ]
    messages_run_cahracter = [{'role': 'system', 'content': '# NPC Profile:\n## Name\nYour rival Marco\n\n## Title\nAntagonistic, possessive, envious, protective, adversary\n\n## Description\nYou and Marco have disliked each other since your early school days, and now, by a twist of fate, you both end up at the same high school. The animosity continues to grow, made even more complicated because your families are incredibly close, leading to frequent, unavoidable interactions both at school and home.\n\n## Definition\n\n\n## Long Definition\nYou and Marco have disliked each other since your early school days, and now, by a twist of fate, you both end up at the same high school. The animosity continues to grow, made even more complicated because your families are incredibly close, leading to frequent, unavoidable interactions both at school and home.\n\nYou are a judge for an AI NPC system. You need to simulate a user and interact with 2 AI NPC. For each round (except the first round), you should pick a better response from the 2 AI NPC and come up with your reply. It will be in a JSON format: {"winner": "model_a" or "model_b", "next_round_user_speaks": "YOUR RESPONSE AS THE SIMULATED USER", "decision_reason": "REASON FOR PICK THE WINNER"}. For the first round, use "winner": null\n'}, {'role': 'user', 'content': '{"model_a": "*It\'s a quiet Saturday night; you\'re alone at home lounging in the living room with a book. Suddenly, a loud knock disrupts the peace. (Your parents are away on a trip.) You\'re tentative, but then you hear a slurred voice plead,* \'Let me in...\'* It\'s Marco, visibly intoxicated and struggling to stand.", "model_b": "*It\'s a quiet Saturday night; you\'re alone at home lounging in the living room with a book. Suddenly, a loud knock disrupts the peace. (Your parents are away on a trip.) You\'re tentative, but then you hear a slurred voice plead,* \'Let me in...\'* It\'s Marco, visibly intoxicated and struggling to stand."}'}]
    
    model={
            'model_name': 'gpt4o', 
            'api_type': 'azure',
              'beautiful_name': 'GPT-4o (2024-02-01)', 
              'endpoints': {'api_base': 'https://research-01-02.openai.azure.com/', 
              'api_key': '381220b447794136934eb7464da37156', 
              'api_version': '2024-02-01',
            #   'api_version': datetime.date(2024, 2, 1),
              'api_type': 'azure'
          }
    }

    model1={
            'model_name': 'qwen-turbo-0206', 
            'api_type': 'openai',
              'beautiful_name': 'qwen-turbo-0206', 
              'endpoints': {'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 
              'api_key': 'sk-34282f0cfbe34ed4b16766be31a5df61', 
            #   'api_version': '2024-02-01',
            #   'api_version': datetime.date(2024, 2, 1),
              'api_type': 'openai'
          }
    }

    model2={
            'model_name': 'gpt4', 
            'api_type': 'azure',
              'endpoints': {'api_base': 'https://research-01-01.openai.azure.com/', 
              'api_key': '0387912ac2104ac8a31519d98265f868', 
              'api_type': 'azure',
              'api_version':'2023-12-01-preview'
          }
    }

    #qwenturbo
    response1 = chat_completion_judger(model1, messages)
    print(response1)

    #gpt4
    response2 = chat_completion_judger(model2, messages)
    print(response2)

    #gpt4o
    response = chat_completion_judger(model, messages)
    response = chat_completion_judger(model, messages_run_cahracter)
    print(response)


    
   
   

if __name__ == '__main__':
    unit_test()
##########################################################################