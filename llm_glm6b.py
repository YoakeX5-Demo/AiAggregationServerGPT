

from transformers import AutoTokenizer, AutoModel
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


async def llm_glm6b(prompt, history, max_length, top_p, temperature):
    global model, tokenizer
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    torch_gc()
    return response

history = []

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).quantize(4).half().cuda()
# tokenizer = AutoTokenizer.from_pretrained(
#     r"C:\Users\23359\.cache\huggingface\hub\models--THUDM--chatglm-6b\snapshots\8b7d33596d18c5e83e2da052d05ca4db02e60620",
#     trust_remote_code=True)
# model = AutoModel.from_pretrained(
#     r"C:\Users\23359\.cache\huggingface\hub\models--THUDM--chatglm-6b\snapshots\8b7d33596d18c5e83e2da052d05ca4db02e60620",
#     trust_remote_code=True).quantize(4).half().cuda()
model.eval()
