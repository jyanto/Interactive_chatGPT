# Interactive AI with open domain
This repository is used to interact with AI from openAI and open domain ability in streamlit app

In this app, user is required to have openAI API key (Please read the [OpenAI API-key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key) )

This app also provides two different LLM models from openAI:
- gpt-3.5-turbo
- text-davinci-003

For using [this app](https://jyanto-interactive-ai-open-domain.streamlit.app/), please go directly to the link
https://jyanto-interactive-ai-open-domain.streamlit.app/

Instruction on generating indexing file to enable open domain:
```
import os
import openai
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

openai.api_key = os.environ["OPENAI_API_KEY"]

documents = SimpleDirectoryReader('PATH_TO_FOLDER').load_data() #read the files inside the folder
index = GPTSimpleVectorIndex(documents) #generate the vector index with llm
index.save_to_disk('index.json') #save the vector index
```

## Reference
1. LlamaIndex https://gpt-index.readthedocs.io/en/latest/guides/primer.html
2. Weng, Lilian. (Oct 2020). How to build an open-domain question answering system? Lil’Log. https://lilianweng.github.io/posts/2020-10-29-odqa/
3. Conversational streamlit app https://github.com/avrabyt
