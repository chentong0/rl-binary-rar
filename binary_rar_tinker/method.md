### 1. Overview

Implement **Binary RAR**, a factuality-oriented reward used for reinforcement learning (RL) training of language models.
The reward ( r(x, y) \in {0, 1} ) measures whether a model response ( y ) contradicts world knowledge given an instruction ( x ).
If no contradiction is detected, the reward is 1; otherwise 0.
This binary signal is used in a **KL-constrained RL objective**.

---

### 2. Components

#### (a) Datastore

* Download and use the **Hugging Face dataset**:

  ```
  tongc-allenai/fact-rl-wildchat-v2
  ```

* Each sample includes the following columns:

  * **id**: string identifier.
  * **messages**: list of dictionaries, each with:

    * `role`: either `"user"` or `"assistant"`.
    * `content`: text content of the message.
  * **ground_truth**: list of JSON-encoded strings; each string contains:

    * `"ground_truth"` – reference factual text.
    * `"docs"` – list of documents used for factual verification.
  * **dataset**: list of strings (unused).

* For each example, treat all `"docs"` fields as **factually correct documents**.
  These serve as the datastore (\mathcal{DS}).

---

### 3. Retrieval

* Use **BM25** to retrieve the most relevant chunks from (\mathcal{DS}).
* For each training pair ((x, y)):

  1. Form a retrieval query by concatenating (x) and (y).
  2. Retrieve the top-(k) (e.g., 5–10) chunks based on BM25 relevance scores.
  3. Denote the retrieved evidence as
     [
     C(x, y) = { d_1, \ldots, d_k }.
     ]
* The retrieval step ensures that factual verification is limited to high-relevance documents.

---

### 4. LM Verifier (Contradiction Detection)

* Use the **Qwen3-32B** model as the LM verifier (also referred to as the “LM judge”).
* The verifier takes as input the tuple ((x, y, C(x, y))).
* The verifier’s task is to detect **contradictions only**, not to confirm full factual coverage.

Example verification prompt:

```
Given the instruction, model response, and retrieved factual documents, 
determine whether any statement in the response contradicts the documents.
Output only:
1  -> if no contradictions are found
0  -> if there is any contradiction
```

* Parse the model’s output and set:
  [
  r(x, y) =
  \begin{cases}
  1 & \text{if output = 1 (no contradiction)}\
  0 & \text{if output = 0 (contradiction detected)}
  \end{cases}
  ]

---

### 5. RL Integration

* Integrate (r(x, y)) into the standard KL-penalized RL objective:
  [
  \max_\theta \mathbb{E}*{y \sim \pi*\theta}[,r(x, y),]

  * \beta, D_{\mathrm{KL}}(\pi_\theta ,|, \pi_{\mathrm{ref}}),
    ]
    where (\pi_{\mathrm{ref}}) is the reference (base) model and (\beta) is the KL-weight.

---

### 6. Efficiency Optimization

#### (a) Pre-Caching Retrieved Documents

* To reduce retrieval cost during training, pre-cache candidate documents for each prompt:

  * For every (x) in the dataset, collect all documents listed in `ground_truth.docs`.
  * Store them as (\mathcal{DS}_{\text{cache}}(x)).
  * During RL training, retrieve (C(x, y)) from this cached subset using BM25 instead of scanning the full datastore.

#### (b) Batched Verification

* Batch multiple ((x, y)) pairs and run the Qwen3-32B verifier in parallel (multi-GPU or asynchronous calls).
* Process all retrieved documents per response in a single forward pass to minimize redundant computation.
