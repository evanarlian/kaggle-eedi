import random

import torch
import torch.nn.functional as F

from utils import late_interaction, manual_late_interaction

# TODO make pytest and/or fuzzing test
queries = F.normalize(torch.randn(5, 10, 256), dim=-1)
documents = F.normalize(torch.randn(7, 30, 256), dim=-1)
manual_queries = []
query_mask = []
for i in range(queries.size(0)):
    n_ones = random.randrange(1, queries.size(1))
    n_zeros = queries.size(1) - n_ones
    mask = torch.tensor([1] * n_ones + [0] * n_zeros)
    manual_queries.append(queries[i][mask.bool()])
    query_mask.append(mask)
query_mask = torch.stack(query_mask)
manual_documents = []
document_mask = []
for i in range(documents.size(0)):
    n_ones = random.randrange(1, documents.size(1))
    n_zeros = documents.size(1) - n_ones
    mask = torch.tensor([1] * n_ones + [0] * n_zeros)
    manual_documents.append(documents[i][mask.bool()])
    document_mask.append(mask)
document_mask = torch.stack(document_mask)
print(queries.size(), documents.size(), query_mask.size(), document_mask.size())
li = late_interaction(queries, documents, query_mask, document_mask)
li2 = manual_late_interaction(manual_queries, manual_documents)
print(li.size(), li2.size())
torch.allclose(li, li2)
