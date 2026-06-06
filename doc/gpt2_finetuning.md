# GPT-2 Fine-Tuning Strategy

The AI Mock Interviewer (V2) uses a fine-tuned **GPT-2 Small** model to dynamically generate context-aware follow-up questions when a candidate provides a mediocre answer.

Here is a detailed breakdown of how the fine-tuning process is implemented in the codebase (specifically within `train/finetune_gpt2.py`).

### 1. The Dataset (`data/gpt2_finetune_data.jsonl`)
A custom dataset of 60 examples was created. Each entry simulates a scenario where a candidate gives a vague or poor answer, mapping it to an ideal follow-up question.
The data is structured as pairs of `prompt` and `completion`:
*   **Prompt:** `"Topic: python\nContext: Explaining python basics\nCandidate Answer: It's something about things working together I think\nFollow-up Question:"`
*   **Completion:** `"Can you provide a specific, concrete example of how python operates in a real-world application?"`

### 2. Freezing the Base Model (Linear Probing)
To prevent destroying GPT-2's core grasp of the English language (and to make training extremely fast on standard hardware), the script uses a technique known as **Head Tuning** or **Linear Probing**:
*   It loads the base `"gpt2"` model from HuggingFace.
*   It loops through all the core transformer blocks (`model.transformer.parameters()`) and freezes them (`requires_grad = False`).
*   **Only the final Language Modeling (LM) head is actively trained.** 

### 3. Tokenization & Formatting
During the data loading phase, the script concatenates the prompt and the completion together, appending the special End-of-Sentence token (`<|endoftext|>`). This teaches the model to read the prompt, generate the follow-up question, and then mathematically "stop" generating text.

### 4. Training Parameters
The training loop utilizes the HuggingFace `Trainer` API with the following hyper-parameters, optimized for small hardware footprints and avoiding CUDA Out-of-Memory errors:
*   **Epochs:** 3
*   **Batch Size:** 4
*   **Gradient Accumulation Steps:** 2 (simulating a larger effective batch size of 8 without eating extra VRAM)
*   **Learning Rate:** `5e-5`
*   **Memory Optimization:** It enables Gradient Checkpointing to save memory during the backward pass, and uses FP16 (mixed precision) if a GPU is available.

### 5. Export and Inference
Finally, the tuned model and its associated tokenizer are exported to the `models/saved/gpt2_finetuned/` directory. During a live interview session, the `InterviewController` attempts to load the model from this directory. If the model or tokenizer fails to load (e.g., due to missing SentencePiece dependencies), the application gracefully falls back to using the base `gpt2` model from the HuggingFace Hub.
