
import llmtune.executor as llmtune



def make_output(raw_output):
    return raw_output.split("### Response:")[1].strip()






llm, llm_config = llmtune.load_llm('llama-7b-4bit', 'llama-7b-4bit.pt')
#llm = llmtune.load_adapter(llm, adapter_path="alpaca-adapter-folder-13b-4bit")
instructions = [
    "what if time travel was possible?",
    "describe a world without technology",
    "imagine a society governed by artificial intelligence",
    "what would happen if humans could communicate with animals?",
    "write a story about a magical adventure",
]
prompts = [
    "Instruction: What if time travel was possible?",
    "Instruction: Describe a world without technology.",
    "Instruction: Imagine a society governed by artificial intelligence.",
    "Instruction: What would happen if humans could communicate with animals?",
    "Instruction: Write a story about a magical adventure."
]
from llmtune.engine.data.alpaca import make_prompt
prompts = [make_prompt(instruction, input_="")  for instruction in instructions]
#prompts = instructions
#prompts = [make_prompt("Continue the daily dialogue.", input_="Hey! How have you been these days?")]
prompts = ["Hey! How have you been these days?"]
for prompt in prompts:
    output = llmtune.generate(
        llm,
        llm_config,
        prompt=prompt,
        min_length=10,
        max_length=200,
        top_p=0.95,
        top_k=50,
        temperature=0.8,
    )
   # output=make_output(output)
    print(output)

