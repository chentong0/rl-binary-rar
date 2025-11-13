import asyncio
import json
import json
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

import chz
import numpy as np
import tinker
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.completers import StopCondition, TinkerMessageCompleter
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.rl.train import AsyncConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer
from .prompts import get_verifier_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# 📚 Dataset Loading and Document Caching
# =============================================================================


@dataclass
class FactDatasetSample:
    """A single sample from the fact-rl-wildchat-v2 dataset."""

    id: str
    messages: list[dict[str, str]]  # list of {"role": str, "content": str}
    ground_truth: list[str]  # list of JSON strings containing "ground_truth" and "docs"
    dataset: list[str]

    def get_prompt_messages(self) -> list[dict[str, str]]:
        """Extract user messages as the prompt (exclude assistant messages)."""
        return [msg for msg in self.messages if msg["role"] == "user"]

    def get_all_docs(self) -> list[str]:
        """Extract all factual documents from ground_truth field."""
        all_docs = []
        for gt_str in self.ground_truth:
            try:
                gt_dict = json.loads(gt_str)
                docs = gt_dict.get("docs", [])
                all_docs.extend(docs)
            except json.JSONDecodeError:
                logger.warning(f"⚠️  Failed to parse ground_truth JSON for sample {self.id}")
                continue
        return all_docs


def load_fact_dataset(split: str = "train") -> list[FactDatasetSample]:
    """
    📥 Load the fact-rl-wildchat-v2 dataset from HuggingFace.
    
    Args:
        split: Dataset split to load ("train" or "test")
    
    Returns:
        List of FactDatasetSample objects
    """
    logger.info(f"📥 Loading dataset chentong00/binary-rar-wildchat-8k (split={split})...")
    dataset = load_dataset("chentong00/binary-rar-wildchat-8k", split=split)
    
    samples = []
    for row in dataset:
        sample = FactDatasetSample(
            id=row["id"],
            messages=row["messages"],
            ground_truth=row.get("ground_truth", []),
            dataset=row.get("dataset", []),
        )
        samples.append(sample)
    
    logger.info(f"✅ Loaded {len(samples)} samples from dataset")
    return samples


# =============================================================================
# 🔍 BM25 Retrieval System
# =============================================================================


class BM25Retriever:
    """BM25-based retriever for finding relevant factual documents."""

    def __init__(
        self,
        documents: list[str],
        tokenizer: Any,
        top_k: int = 5,
        chunk_size: int = 512,
    ):
        """
        Initialize BM25 retriever with chunking.
        
        Args:
            documents: List of factual documents to index
            tokenizer: Tokenizer to use for chunking documents
            top_k: Number of top chunks to retrieve
            chunk_size: Maximum number of tokens per chunk
        """
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        
        # Chunk and tokenize documents
        logger.info(f"🔨 Building BM25 index for {len(documents)} documents with chunk_size={chunk_size}...")
        self.chunks: list[str] = []
        self.chunk_tokenized_list: list[list[str]] = []
        
        for doc_obj in documents:
            doc_title = doc_obj["title"]
            doc_text = doc_obj["text"]
            doc_text = doc_title + "\n\n" + doc_text
            
            # Tokenize the document
            doc_tokens = self.tokenizer.encode(doc_text, max_length=None, truncation=False, add_special_tokens=False)
            
            # Split into chunks of chunk_size tokens
            for i in range(0, len(doc_tokens), chunk_size):
                chunk_token_ids = doc_tokens[i : i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_token_ids)
                self.chunks.append(chunk_text)
                
                # For BM25, use simple word tokenization of the chunk
                chunk_tokenized = chunk_text.lower().split()
                self.chunk_tokenized_list.append(chunk_tokenized)
        
        self.bm25 = BM25Okapi(self.chunk_tokenized_list)
        logger.info(f"✅ BM25 index built successfully with {len(self.chunks)} chunks")

    def retrieve(self, query: str) -> list[str]:
        """
        Retrieve top-k most relevant document chunks for a query.
        
        Args:
            query: Query string (concatenation of instruction and response)
        
        Returns:
            List of top-k most relevant chunk strings
        """
        query_tokenized = query.lower().split()
        scores = self.bm25.get_scores(query_tokenized)
        top_indices = np.argsort(scores)[::-1][: self.top_k]
        return [self.chunks[i] for i in top_indices if i < len(self.chunks)]


# =============================================================================
# 🧠 Verifier for Contradiction Detection
# =============================================================================


class ContradictionVerifier:
    """LM-based verifier using to detect contradictions."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer_name: str,
        model_name: str,
        max_tokens: int = 10,
    ):
        """
        Initialize the verifier.
        
        Args:
            sampling_client: Tinker sampling client for Qwen3-32B
            renderer_name: Name of the renderer to use
            model_name: Model name for tokenizer
            max_tokens: Maximum tokens to generate
        """
        self.renderer = renderers.get_renderer(renderer_name, get_tokenizer(model_name))
        self.completer = TinkerMessageCompleter(sampling_client, self.renderer, max_tokens)
        self.first_call = False  # Track first call in batch for logging

    async def verify(self, instruction: str, response: str, documents: list[str]) -> float:
        """
        Verify if response contradicts the documents.
        
        Args:
            instruction: User instruction/prompt
            response: Model's response to verify
            documents: Retrieved factual documents
        
        Returns:
            Binary reward: 1.0 if no contradiction, 0.0 if contradiction detected
        """
        # Format documents
        docs_text = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])
        
        # Create verification prompt
        prompt_text = get_verifier_prompt(
            instruction=instruction,
            response=response,
            documents=docs_text,
        )

        # Get verification from LM
        messages = [{"role": "user", "content": prompt_text}]
        try:
            output_message = await self.completer(messages)
            output_text = output_message["content"].strip()
            
            res_obj = json.loads(output_text.strip("```json").strip("```"))
            score = res_obj.get("SCORE", 0)
            reasoning = res_obj.get("REASONING", "")
            
            if score == 1:
                logger.debug(f"✅ Verification: NO contradiction detected - {reasoning}")
            else:
                logger.debug(f"❌ Verification: Contradiction detected - {reasoning}")
            
            # Log output for first call in batch
            if self.first_call:
                logger.info("=" * 80)
                logger.info("📝 VERIFIER FIRST CALL IN BATCH - INPUT:")
                logger.info("=" * 80)
                logger.info(f"{prompt_text}")
                logger.info("=" * 80)
                logger.info("📝 VERIFIER FIRST CALL IN BATCH - OUTPUT:")
                logger.info("=" * 80)
                logger.info(f"{output_text}")
                logger.info("=" * 80)
                logger.info(f"Parsed score: {score}")
                logger.info(f"Reasoning: {reasoning}")
                logger.info("=" * 80 + "\n")
                self.first_call = False  # Disable logging for subsequent calls
            
            return float(score)
        except Exception as e:
            try:
                first_100_chars = output_text[:100]
                logger.warning(f"⚠️  Verification failed with error: {e}, output_text: {first_100_chars}..., returning 0.0")
            except Exception as e:
                logger.warning(f"⚠️  Verification failed with error: {e}, returning 0.0")
            if self.first_call:
                logger.info("=" * 80)
                logger.info("📝 VERIFIER FIRST CALL IN BATCH - INPUT:")
                logger.info("=" * 80)
                logger.info(f"{prompt_text}")
                logger.info("=" * 80)
                logger.info("📝 VERIFIER FIRST CALL IN BATCH - OUTPUT:")
                logger.info("=" * 80)
                logger.info(f"Reward: 0.0")
                logger.info("=" * 80 + "\n")
                self.first_call = False
            return 0.0


# =============================================================================
# 🎮 Factuality Environment
# =============================================================================


class FactualityEnv(Env):
    """
    Environment for factuality-based RL training.
    
    Each episode:
    1. Agent receives a user prompt
    2. Agent generates a response
    3. Response is verified against retrieved documents
    4. Binary reward (0 or 1) is returned
    """

    def __init__(
        self,
        prompt_messages: list[dict[str, str]],
        policy_renderer: renderers.Renderer,
        bm25_retriever: BM25Retriever,
        verifier: ContradictionVerifier,
    ):
        """
        Initialize factuality environment.
        
        Args:
            prompt_messages: User prompt messages
            policy_renderer: Renderer for policy
            bm25_retriever: BM25 retriever for documents
            verifier: Contradiction verifier
        """
        self.prompt_messages = prompt_messages
        self.policy_renderer = policy_renderer
        self.bm25_retriever = bm25_retriever
        self.verifier = verifier
        # reasoning output
        self.reasoning_content: str | None = None
        self.content: str | None = None
        self.generated_response: str | None = None

    @property
    def stop_condition(self) -> StopCondition:
        return self.policy_renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Return the initial prompt for the agent."""
        return (
            self.policy_renderer.build_generation_prompt(self.prompt_messages),
            self.stop_condition,
        )

    async def step(self, action: Action) -> StepResult:
        """
        Process agent's action and compute reward.
        
        Args:
            action: Token IDs generated by the policy
        
        Returns:
            StepResult with binary reward and metrics
        """
        # Parse the response
        response_message, parse_success = self.policy_renderer.parse_response(action)
        response_text = response_message["content"]
        self.generated_response = response_text

        match = re.search(r'<think>(.*?)</think>(.*)', response_text, re.DOTALL)
        if match:
            self.reasoning_content = match.group(1)
            self.content = match.group(2)
        else:
            self.reasoning_content = None
            self.content = response_text
        
        # If parsing failed, return 0 reward
        if not parse_success:
            logger.debug(f"⚠️  Response parsing failed")
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"parse_success": 0, "contradiction_detected": 1},
            )
        
        # Retrieve documents
        # Concatenate instruction and response for retrieval
        instruction_text = "\n\n".join([msg["content"] for msg in self.prompt_messages])
        retrieval_query = self.content
        retrieved_docs = self.bm25_retriever.retrieve(retrieval_query)
        
        logger.debug(f"🔍 Retrieved {len(retrieved_docs)} documents")

        # Verify response against documents
        reward = await self.verifier.verify(instruction_text, self.content, retrieved_docs)
        
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "parse_success": 1,
                "contradiction_detected": 1 - reward,
                "num_retrieved_docs": len(retrieved_docs),
            },
        )


# =============================================================================
# 🎯 Environment Group Builder
# =============================================================================


@dataclass(frozen=True)
class FactualityEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of factuality environments for RL training."""

    prompt_messages: list[dict[str, str]]
    policy_renderer: renderers.Renderer
    bm25_retriever: BM25Retriever
    verifier: ContradictionVerifier
    group_size: int

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments with the same prompt."""
        return [
            FactualityEnv(
                prompt_messages=self.prompt_messages,
                policy_renderer=self.policy_renderer,
                bm25_retriever=self.bm25_retriever,
                verifier=self.verifier,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        """Group rewards are already computed in step(), so return zeros."""
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["factuality", "binary_rar"]


# =============================================================================
# 📊 RL Dataset
# =============================================================================


class FactualityRLDataset(RLDataset):
    """Dataset for factuality RL training."""

    def __init__(
        self,
        samples: list[FactDatasetSample],
        policy_renderer: renderers.Renderer,
        batch_size: int,
        group_size: int,
        verifier_builder: Callable[[], ContradictionVerifier],
        retriever_tokenizer: Any,
        bm25_top_k: int = 5,
        bm25_chunk_size: int = 512,
    ):
        """
        Initialize RL dataset.
        
        Args:
            samples: List of fact dataset samples
            policy_renderer: Renderer for policy
            batch_size: Number of prompts per batch
            group_size: Number of trajectories per prompt
            verifier_builder: Function to build a verifier
            retriever_tokenizer: Tokenizer for chunking documents
            bm25_top_k: Number of document chunks to retrieve with BM25
            bm25_chunk_size: Number of tokens per chunk
        """
        self.samples = samples
        self.policy_renderer = policy_renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.verifier_builder = verifier_builder
        self.bm25_top_k = bm25_top_k
        self.bm25_chunk_size = bm25_chunk_size
        
        # Pre-cache BM25 retrievers for each sample (efficiency optimization)
        logger.info(f"🚀 Pre-caching BM25 retrievers for {len(samples)} samples...")
        self.retrievers: dict[str, BM25Retriever] = {}
        for sample in samples:
            docs = sample.get_all_docs()
            if len(docs) > 0:
                self.retrievers[sample.id] = BM25Retriever(
                    docs,
                    tokenizer=retriever_tokenizer,
                    top_k=bm25_top_k,
                    chunk_size=bm25_chunk_size,
                )
            else:
                logger.warning(f"⚠️  Sample {sample.id} has no documents, skipping")
        logger.info(f"✅ Pre-cached {len(self.retrievers)} BM25 retrievers")

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.samples))
        batch_samples = self.samples[start_idx:end_idx]
        
        verifier = self.verifier_builder()
        if index == 0:
            verifier.first_call = True
        
        env_group_builders = []
        for sample in batch_samples:
            # Skip samples without cached retrievers
            if sample.id not in self.retrievers:
                continue
            
            builder = FactualityEnvGroupBuilder(
                prompt_messages=sample.get_prompt_messages(),
                policy_renderer=self.policy_renderer,
                bm25_retriever=self.retrievers[sample.id],
                verifier=verifier,
                group_size=self.group_size,
            )
            env_group_builders.append(builder)
        
        return env_group_builders

    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.samples) + self.batch_size - 1) // self.batch_size


# =============================================================================
# ⚙️  CLI Configuration
# =============================================================================


@chz.chz
class CLIConfig:
    """Command-line configuration for Binary RAR training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str | None = "qwen3"
    lora_rank: int = 32
    load_checkpoint_path: str | None = None

    # Verifier configuration
    verifier_model_name: str = "Qwen/Qwen3-8B"
    verifier_renderer_name: str | None = "qwen3_disable_thinking"
    verifier_max_tokens: int = 512

    # Training hyperparameters
    group_size: int = 16
    groups_per_batch: int = 16
    learning_rate: float = 1e-5
    max_tokens: int = 4096
    kl_penalty_coef: float = 0.001

    # BM25 configuration
    bm25_top_k: int = 8
    bm25_chunk_size: int = 512

    # Dataset configuration
    max_samples: int | None = None  # Limit samples for debugging

    # Number of optimizer steps per training iteration
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evaluation and checkpointing
    eval_every: int = 100
    save_every: int = 100

    # Async RL configuration (for better efficiency)
    use_async_rl: bool = True  # Enable async off-policy training
    max_steps_off_policy: int = 2  # Max steps off-policy before discarding samples
    async_groups_per_batch: int | None = None  # Groups per batch for async (defaults to groups_per_batch if None)

    # Data filtering
    remove_constant_reward_groups: bool = True  # Ignore prompts where all responses get the same reward

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


# =============================================================================
# 🚀 Main Training Function
# =============================================================================


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    logger.info("🔬 Starting Binary RAR Training for Hallucination Reduction")
    logger.info(f"📊 Model: {cli_config.model_name}")
    logger.info(f"🧠 Verifier: {cli_config.verifier_model_name}")
    logger.info(f"🎯 Group size: {cli_config.group_size}")
    logger.info(f"📦 Batch size: {cli_config.groups_per_batch}")
    if cli_config.remove_constant_reward_groups:
        logger.info(f"🔍 Filtering enabled: Will skip prompts where all responses get the same reward")

    # Get renderer names
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    verifier_renderer_name = cli_config.verifier_renderer_name or model_info.get_recommended_renderer_name(
        cli_config.verifier_model_name
    )

    # Create log path
    model_name_safe = cli_config.model_name.replace("/", "-")
    run_name = f"binary-rar-{model_name_safe}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.group_size}group-{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    
    if cli_config.log_path is not None:
        log_path = f"{cli_config.log_path}/{run_name}"
    else:
        log_path = f"/tmp/tinker-examples/binary_rar/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = f"{cli_config.wandb_name}-{run_name}"
    else:
        wandb_name = run_name

    # Initialize service client for verifier
    service_client = tinker.ServiceClient(base_url=cli_config.base_url)
    verifier_sampling_client = service_client.create_sampling_client(
        base_model=cli_config.verifier_model_name
    )
    
    logger.info(f"✅ Created verifier sampling client for {cli_config.verifier_model_name}")

    # Create verifier builder with access to sampling client
    def verifier_builder() -> ContradictionVerifier:
        return ContradictionVerifier(
            sampling_client=verifier_sampling_client,
            renderer_name=verifier_renderer_name,
            model_name=cli_config.verifier_model_name,
            max_tokens=cli_config.verifier_max_tokens,
        )

    # Create dataset builder with verifier
    # We need to modify the dataset to accept verifier_builder properly
    # Let's create a custom dataset class that has the verifier builder
    
    # Load dataset samples
    logger.info("📥 Loading dataset...")
    samples = load_fact_dataset(split="train")
    if cli_config.max_samples is not None:
        samples = samples[: cli_config.max_samples]
        logger.info(f"🔬 Limited to {len(samples)} samples")
    
    # Get policy renderer
    policy_renderer = renderers.get_renderer(
        renderer_name, get_tokenizer(cli_config.model_name)
    )
    logger.info(f"✅ Loaded policy renderer for {cli_config.model_name}")
    
    # Get retriever tokenizer for BM25 chunking
    retriever_tokenizer = get_tokenizer(cli_config.verifier_model_name)
    retriever_tokenizer.model_max_length = 0
    logger.info(f"✅ Loaded retriever tokenizer for {cli_config.verifier_model_name}")
    
    # Create dataset
    rl_dataset = FactualityRLDataset(
        samples=samples,
        policy_renderer=policy_renderer,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        verifier_builder=verifier_builder,
        retriever_tokenizer=retriever_tokenizer,
        bm25_top_k=cli_config.bm25_top_k,
        bm25_chunk_size=cli_config.bm25_chunk_size,
    )
    
    # Create a simple dataset builder that returns the dataset
    @chz.chz
    class SimpleDatasetBuilder(RLDatasetBuilder):
        async def __call__(self) -> tuple[RLDataset, None]:
            return rl_dataset, None
    
    dataset_builder = SimpleDatasetBuilder()

    # Create async config if enabled
    async_config = None
    if cli_config.use_async_rl:
        async_groups = cli_config.async_groups_per_batch or cli_config.groups_per_batch
        async_config = AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=async_groups,
        )
        logger.info(f"🚀 Async RL enabled: max_steps_off_policy={cli_config.max_steps_off_policy}, groups_per_batch={async_groups}")

    # Create full config
    config = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        evaluator_builders=[],  # Could add evaluators here
        async_config=async_config,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
    )

    logger.info(f"🔍 Config: {config}")

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    logger.info(f"🎯 Starting RL training...")
    logger.info(f"📁 Log path: {log_path}")
    logger.info(f"📊 Total batches: {len(rl_dataset)}")

    # Run training
    await train.main(config)

    logger.info("🎉 Training completed successfully!")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))

