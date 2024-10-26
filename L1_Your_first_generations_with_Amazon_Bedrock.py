#!/usr/bin/env python
# coding: utf-8

# # Text generations with Amazon Bedrock

# This demo showcases Amazon Bedrock's text generation capabilities. We'll explore customizing model prompts and outputs to tailor the responses for your specific needs.

# ### Import all needed packages

# Boto3 is the official AWS SDK for Python, enabling seamless interaction with AWS services. It simplifies resource management, API calls, and automation across the entire AWS ecosystem.

# In[1]:


import boto3
import json


# ### Setup the Bedrock runtime

# The `boto3.client()` function creates a low-level client for the 'bedrock-runtime' AWS service, allowing programmatic access to the Bedrock service in the 'us-west-2' AWS region.

# In[2]:


bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')


# In[20]:


prompt = "Write a 5 sentence summary of Kerala "


# Set up `kwargs` dictionary with the necessary parameters to invoke the Bedrock model. Specifically:
# 
# - `"modelId"`: Specifies the ID of the Bedrock model to be used, in this case, "amazon.titan-text-lite-v1".
# - `"contentType"` and `"accept"`: Set the content type and expected response format.
# - `"body"`: Contains the input text (stored in the `prompt` variable) as a JSON-encoded string.
# 
# These parameters are then passed as keyword arguments to the `invoke_model()` method in the previous step, allowing the Bedrock service to generate text based on the provided prompt.

# In[21]:


kwargs = {
    "modelId": "amazon.titan-text-lite-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt
        }
    )
}


# invoke_model() method from the Bedrock runtime client created earlier to invoke a Bedrock model with the specified keyword arguments (**kwargs). This allows you to leverage the text generation capabilities of the Bedrock service.

# In[22]:


response = bedrock_runtime.invoke_model(**kwargs)


# In[23]:


response


# In[24]:


response_body = json.loads(response.get('body').read())


# In[25]:


print(response_body)


# In[26]:


print(json.dumps(response_body, indent=4))


# In[27]:


print(response_body['results'][0]['outputText'])


# ### Generation Configuration

# In[40]:


prompt = "Write a summary of Kerala. In less than 500 words"


# In[41]:


kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}


# In[42]:


response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)


# In[43]:


print(json.dumps(response_body, indent=4))


# In[44]:


kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1000,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}


# In[45]:


response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)


# In[46]:


print(json.dumps(response_body, indent=4))


# ### Working with other type of data

# In[47]:


from IPython.display import Audio


# In[48]:


audio = Audio(filename="dialog.mp3")
display(audio)


# In[49]:


with open('transcript.txt', "r") as file:
    dialogue_text = file.read()


# In[50]:


print(dialogue_text)


# In[51]:


prompt = f"""The text between the <transcript> XML tags is a transcript of a conversation. 
Write a short summary of the conversation.

<transcript>
{dialogue_text}
</transcript>

Here is a summary of the conversation in the transcript:"""


# In[52]:


print(prompt)


# In[53]:


kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0,
                "topP": 0.9
            }
        }
    )
}


# In[54]:


response = bedrock_runtime.invoke_model(**kwargs)


# In[55]:


response_body = json.loads(response.get('body').read())
generation = response_body['results'][0]['outputText']


# In[56]:


print(generation)

