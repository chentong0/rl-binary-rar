import asyncio
import concurrent.futures
import time
from typing import List, Any
from tqdm import tqdm
import logging

async def run_in_executor(executor, fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, fn, *args)

async def batch_call_async_internal(fn, args_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [run_in_executor(executor, fn, *args) for args in args_list]
        return await asyncio.gather(*tasks, return_exceptions=True)

def batch_call_async(fn, args_list):
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # If no event loop is running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(batch_call_async_internal(fn, args_list))
    except RuntimeError:  # If an event loop is already running
        import nest_asyncio
        nest_asyncio.apply()  # Allows reusing the running event loop in Jupyter
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(batch_call_async_internal(fn, args_list))

def batch_chat_complete_process_batch(client, messages_batch, max_retries=5, *args, **kwargs):
    def chat_complete_fn(messages):
        for k in range(max_retries):
            try:
                time.sleep(0.1)
                return client.chat.completions.create(
                    messages=messages,
                    *args,
                    **kwargs
                )
                # wait for 0.1 seconds
            except Exception as e:
                if k == max_retries - 1:  # Last attempt
                    raise e
                time.sleep(2 ** k)  # Exponential backoff
                logging.warning(f"Error: {e}. Retrying...")
        logging.warning(f"Error: failed to complete {messages}")
    return batch_call_async(chat_complete_fn, [(messages,) for messages in messages_batch])

def batch_chat_complete(client, messages_list, batch_size=512, max_retries=5, *args, **kwargs):
    all_responses = []
    for i in tqdm(range(0, len(messages_list), batch_size), desc="Processing chat completion"):
        batch = messages_list[i:i + batch_size]
        batch_responses = batch_chat_complete_process_batch(client, batch, max_retries, *args, **kwargs)
        all_responses.extend(batch_responses)
    return all_responses

# def batch_complete(client, prompt_list, *args, **kwargs):
#     def chat_complete_fn(prompt):
#         return client.completions.create(
#             prompt=prompt,
#             *args,
#             **kwargs
#         )
#     responses = batch_call_async(chat_complete_fn, [(prompt,) for prompt in prompt_list])
#     return responses
