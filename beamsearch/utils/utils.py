import torch
import torch.nn.functional as F
import random


def get_top_p_indices(sm_tensor, p):
    probs, indices = torch.sort(sm_tensor, descending=True)
    cum_sum = torch.cumsum(probs, dim=-1)
    mask = cum_sum > p
    mask[..., 1:] = mask[..., :-1].clone() #shift right by one
    mask[..., 0] = False

    return indices[mask], indices[~mask]

def get_inputs(str_input, tokenizer, llm_bos_id=1):
    inputs = tokenizer(str_input, add_special_tokens=False, return_tensors="pt")["input_ids"]-2
    inputs = torch.cat((torch.tensor([[llm_bos_id]]), inputs), dim=-1)[0]
    return inputs

# MEAN_VECTOR = torch.tensor()TODO
def get_loss(input_tokens, bc):
    with torch.no_grad():
        input_tokens = input_tokens.unsqueeze(-1).to(bc.embedding_model.device)
        outputs = bc.embedding_model(input_tokens)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)

        cos_target = torch.ones(1).to(bc.device)
        loss = F.cosine_embedding_loss(F.normalize(sentence_embedding, p=2.0, dim=-1), bc.target, cos_target)

    return loss


# def project(v, t):
#     """Projects V onto T's normal plane"""
#     v = v - ((torch.dot(v, t) / (torch.norm(t)**2+1e-8)) * t)
#     return v

# def gradient_surgery(emb_grads, reg_grads, reg_grads_constant=0.001):
#     for i in range(seq_len):
#         if torch.dot(emb_grads[i], reg_grads[i]) < 0:
#             priority = random.choices(["emb", "reg"], weights=[1./reg_grads_constant, 1])[0]
#             if priority == "emb":
#                 reg_grads[i] = project(reg_grads[i], emb_grads[i])
#             elif priority == "reg":
#                 emb_grads[i] = project(emb_grads[i], reg_grads[i])
#             else:
#                 raise Exception("Projection error")

#     return emb_grads + reg_grads * reg_grads_constant