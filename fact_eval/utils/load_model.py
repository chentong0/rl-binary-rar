import os

def load_model(model_name):
    if model_name.startswith("azure-openai::"):
        from openai import AzureOpenAI
        assert os.getenv("AZURE_OPENAI_API_VERSION", None) != None
        assert os.getenv("AZURE_OPENAI_ENDPOINT", None) != None
        assert os.getenv("AZURE_OPENAI_API_KEY", None) != None
        client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        return {"client": client, "tokenizer": None, "llm": None}
    elif model_name.startswith("openai::"):
        from openai import OpenAI
        assert os.getenv("OPENAI_API_KEY", None) != None
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        return {"client": client, "tokenizer": None, "llm": None}
    elif model_name.startswith("vllm-openai::"):
        from openai import OpenAI
        assert os.getenv("VLLM_OPENAI_BASE_URL", None) != None
        client = OpenAI(
            base_url=os.getenv("VLLM_OPENAI_BASE_URL"),
            api_key="None",
        )
        return {"client": client, "tokenizer": None, "llm": None}
    else:
        # import torch
        from vllm import LLM
        from transformers import AutoTokenizer

        # get the available gpu memory, and the model should take 20gb
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        # available_memory = total_memory - torch.cuda.memory_allocated()
        # Calculate memory utilization based on available memory vs desired 20GB
        # desired_memory = 20 * 1024 * 1024 * 1024  # 20GB in bytes
        # gpu_memory_utilization = min(1.0, desired_memory / total_memory)

        llm = LLM(
            model=model_name, 
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return {"client": None, "tokenizer": tokenizer, "llm": llm}
