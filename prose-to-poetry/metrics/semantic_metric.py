import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

tokenizer_sent = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
model_sent = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2").cuda()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def encode_sent(texts, batch_size=8):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        encoded = tokenizer_sent(batch, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = model_sent({k: v.cuda() for k, v in encoded.items()})

        embeddings = mean_pooling(model_output, encoded['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

def embedding_sim_score(pred, emb_prose):
    emb_poem = encode_sent(pred)
    cos_sim = (emb_prose * emb_poem).sum(dim=1).cpu()
    return cos_sim

def make_semantic_reward(coef):
    def sem_reward(completions, emb_prose=None, **kwargs):
        scores = embedding_sim_score(completions, emb_prose)
        return (coef * scores).tolist()
    
    return sem_reward