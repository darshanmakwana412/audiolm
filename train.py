import torch
import numpy as np
import pickle
from models import GPT
import wandb
from tqdm import tqdm
import soundfile as sf
import torchaudio
from speechtokenizer import SpeechTokenizer

CONFIG = {
    "vocab_size": 1024,
    "context_length": 2048,
    "emb_dim": 6 * 64,
    "n_heads": 6,
    "n_layers": 6,
    "head_dim": 64,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "mlp_hidd_dim": 4 * 6 * 64,
    "verbose": True,
    "batch_size": 32,
    "device": "cuda:6",
    "lr": 1e-4,
    "steps": 20000,
    "eval_iters": 100
}
audio_device = "cuda:7"

config_path = './SpeechTokenizer/config.json'
ckpt_path = './SpeechTokenizer/SpeechTokenizer.pt'
audio_model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
audio_model.eval()
audio_model.to(audio_device)

data = np.load("./data/semantic_tokens.npy")

# create train/val split
N = int(data.shape[0] * 0.95)
train_data = data[:N]
val_data = data[N:]
print(f"Total number of tokens used for training: {len(train_data)/1e6:.2f} Million")
print(f"Total number of tokens used for validation: {len(val_data)/1e6:.2f} Million")

# A simple dataloader based on random sampling
def get_batch(split="train"):

    if split == "train":
        batch_data = train_data
    else:
        batch_data = val_data
    
    idx = np.random.randint(0, batch_data.shape[0] - CONFIG["context_length"], size=(CONFIG["batch_size"]))
    x = torch.from_numpy((np.array([batch_data[i: i + CONFIG["context_length"]] for i in idx])).astype(np.int64))
    y = torch.from_numpy((np.array([batch_data[i + 1: i + CONFIG["context_length"] + 1] for i in idx])).astype(np.int64))

    if CONFIG["device"][:4] == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(CONFIG["device"], non_blocking=True), y.pin_memory().to(CONFIG["device"], non_blocking=True)
    else:
        x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
    return x, y

model = GPT(CONFIG)
model.to(CONFIG["device"])

step = 0
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

wandb.init(project="audio", config=CONFIG)

model = torch.compile(model)

while step < model.cfg["steps"]:
    
    if step % 100 == 0:
        
        model.eval()
        torch.cuda.empty_cache()
        
        val_losses = []
        for _ in range(CONFIG["eval_iters"]):
            x, y = get_batch("val")
            with torch.no_grad():
                logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
            val_losses.append(loss.item())
        
        avg_loss = sum(val_losses)/len(val_losses)
        print(f"Val loss at step {step}: {avg_loss}")
        wandb.log({"Val Loss": avg_loss})
        
        # Log some sample from the model
        x, y = get_batch("val")
        gen_idxs = model.generate(x[:1, :512], CONFIG["context_length"])
        audio_tokens = gen_idxs.unsqueeze(0)
        wav = audio_model.decode(audio_tokens[:, :, 512:].to(audio_device))
        torchaudio.save(f"./samples/step_{step}.mp3", wav.squeeze(0).detach().cpu(), audio_model.sample_rate)
        
#         codes = gen_idxs[0].view(-1, 4).transpose(0, 1)[None, None]
#         audio_values = audio_model.decode(codes.to(audio_device), [None])[0]
#         values = audio_values.detach().cpu().numpy()[0, 0]
        
#         sf.write(f"./samples/sample_{step}.wav", values, processor.sampling_rate, 'PCM_24')
    
    model.train()
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    
    x, y = get_batch()
    
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
    print(step, loss.item())
    
    wandb.log({"Train Loss": loss.item()})
    
    loss.backward()
    optimizer.step()
        
    if step % 1000 == 0:
        torch.save({'model': model.state_dict(), 'config': CONFIG}, f"./models/gpt_{step}.pt")
        
    step += 1

wandb.finish()

torch.save({'model': model.state_dict(), 'config': CONFIG}, f"./models/gpt_final.pt")