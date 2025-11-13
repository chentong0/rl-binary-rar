from typing import Dict, Any, List, Tuple
import vllm

class BaseScorer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _generate(self, model_name: str, prompt_list: List[str]) -> List[str]:
        llm = vllm.LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        results = llm.generate(prompt_list, sampling_params=vllm.SamplingParams(
            temperature=0.0, max_tokens=1024
        ))
        response_list = [result.outputs[0].text for result in results]
        return response_list

    def _score(self, prompt_list: List[str], response_list: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _aggregate(self, per_instance_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def run_eval(self, model_name: str, data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        prompt_list = [item["question"] for item in data]
        response_list = self._generate(model_name, prompt_list)
        per_instance_results = self._score(prompt_list, response_list)
        per_instance_results = [{**item, **per_instance_results[i]} for i, item in enumerate(data)]
        aggregate_results = self._aggregate(per_instance_results)
        return aggregate_results, per_instance_results
