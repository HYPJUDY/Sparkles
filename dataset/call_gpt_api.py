import openai
from time import sleep
import os
# Fetch the API environment variables
api_key = os.getenv("OPENAI_API_KEY")  # export OPENAI_API_KEY="your-api-key-here"
organization = os.getenv("OPENAI_ORGANIZATION")  # export OPENAI_ORGANIZATION="your-organization-here"
api_base = os.getenv("OPENAI_API_BASE")  # export OPENAI_API_BASE="https://your-endpoint.openai.com"
# Set the API key (mandatory)
if api_key is not None:
    openai.api_key = api_key
else:
    print("You have not set OpenAI API key, "
          "which is required for SparklesDialogue generation or SparklesEval evaluation.")

# Set the organization if it's specified
if organization is not None:
    openai.organization = organization

if api_base is not None:
    openai.api_base = api_base

def openai_chat_create(user_content, temperature=1.0, max_tokens=2048, top_p=1,
                       frequency_penalty=0.0, presence_penalty=0.0, model="gpt-4"):
    try:
        # For error of "Error communicating with OpenAI: HTTPSConnectionPool(host= Max retries exceeded with url:"
        # it seems that embedding the API key in the code works...
        # openai.api_key = "your-api-key-here"
        if api_base is not None:
            # engine="gpt-4", engine="gpt-35-turbo"

            # openai.api_base = "https://your-endpoint.openai.azure.com/"

            openai.api_type = "azure"
            openai.api_version = "2023-07-01-preview"
            if model == "gpt-3.5-turbo":
                model = "gpt-35-turbo"
            arg = {"engine": model}
        else:
            # model="gpt-4", model="gpt-3.5-turbo"
            # openai.organization = "your-organization-here"
            arg = {"model": model}

        # https://platform.openai.com/docs/api-reference/chat/create
        response = openai.ChatCompletion.create(
            **arg,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        message = response["choices"][0]["message"]
        return message["content"]
    except Exception as e:
        print(e)
        sleep(10)
        return openai_chat_create(user_content, temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                                  frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

