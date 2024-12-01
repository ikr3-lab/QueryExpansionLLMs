import sys
import time

import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
from transformers import pipeline, Conversation
from transformers import BitsAndBytesConfig
from pyterrier.measures import *
import pyterrier as pt
pt.init()

tokenizerPT = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
def strip_markup(text):
    return " ".join(tokenizerPT.getTokens(text))

modelname = "meta-llama/Meta-Llama-3-8B-Instruct"
print(modelname)
modelnameinfile = "Meta-Llama-3-8B-Instruct"

access_token = "YOUR ACCESS TOKEN HERE"
tokenizer = AutoTokenizer.from_pretrained(modelname,  token = access_token)

model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto", token = access_token, torch_dtype=torch.bfloat16)
print(model)

name_dataset = "irds:beir/scifact/test"
name_folder = "scifactindex_testLLama3_8b"

test_dataset = pt.get_dataset(name_dataset)
indexer = pt.index.IterDictIndexer(pathindex+name_folder, overwrite=True, meta={'docno': 200})
indexref = indexer.index(test_dataset.get_corpus_iter())
index = pt.IndexFactory.of(indexref)

BM25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.b":0.75, "bm25.k_1":1.2, "bm25.k_3":8.0})
BM25_bo1 = pt.BatchRetrieve(index, wmodel="BM25", controls={"qe":"on", "qemodel" : "Bo1"})
BM25_bo2 = pt.BatchRetrieve(index, wmodel="BM25", controls={"qe":"on", "qemodel" : "Bo2"})
BM25_klc = pt.BatchRetrieve(index, wmodel="BM25", controls={"qe":"on", "qemodel" : "KLCorrect"})

data_query_original = test_dataset.get_topics()
qrels = test_dataset.get_qrels()

def template_document_zs(query):
    return f'''Write a passage that answers the given query: {query}'''

def template_keywords_zs(query):
    return f'''Write a list of keywords for the following query: {query}'''

def template_cot_zs(query):
    return f'''Answer the following query:\n{query}\nGive the rationale before answering'''

queries = data_query_original["query"]
new_queries_docs = []
new_queries_keywords = []
new_queries_cot = []

finpass = open(pathindex+name_folder+"/generated_passage_"+modelnameinfile+".txt", "w")
finkey = open(pathindex+name_folder+"/generated_keywords_"+modelnameinfile+".txt", "w")
fincot = open(pathindex+name_folder+"/generated_cot_"+modelnameinfile+".txt", "w")

#cleaning function
def cleanOutput(query):
    return strip_markup(query.strip())


def querymodel(messageINPUT):
    messages = [
       {"role": "user", "content": messageINPUT.strip()},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
     input_ids,
     max_new_tokens=256,
     eos_token_id=terminators,
     do_sample=True,
     temperature=0.6,
     top_p=0.9,
     pad_token_id=tokenizer.eos_token_id
    )
    response = outputs[0][input_ids.shape[-1]:]
    answer =tokenizer.decode(response, skip_special_tokens=True).replace("\n", " ").strip()
    return answer

n_query = 5
for query_orig in queries:
    query_orig = query_orig.strip()

    document_template = template_document_zs(query_orig)
    document = querymodel(document_template)
    finpass.write(document)
    finpass.write("\n")


    keywords_template = template_keywords_zs(query_orig)
    keywords = querymodel(keywords_template)
    finkey.write(keywords)
    finkey.write("\n")


    cot_template = template_cot_zs(query_orig)
    cot = querymodel(cot_template)
    fincot.write(cot)
    fincot.write("\n")


    query_ntimes = (" " + query_orig) * n_query
    query_ntimes = query_ntimes.strip()

    document = cleanOutput(document)
    document = query_ntimes+" "+document
    new_queries_docs.append(document)

    keywords = cleanOutput(keywords)
    keywords = query_ntimes+" "+keywords
    new_queries_keywords.append(keywords)

    cot = cleanOutput(cot)
    cot = query_ntimes+" "+cot
    new_queries_cot.append(cot)

finpass.close()
finkey.close()
fincot.close()

print("BASELINE ORIGINAL QUERY")
out = pt.Experiment(
    [BM25, BM25_bo1, BM25_bo2, BM25_klc],
    data_query_original,
    qrels,
    eval_metrics=["map", "recall_1000"],
    names=["BM25", "BM25_bo1","BM25_bo2","BM25_klc"]
)
print(out)

print("Q2D/ZS")
data_query_original["query"]=new_queries_docs
out = pt.Experiment(
    [BM25, BM25_bo1, BM25_bo2, BM25_klc],
    data_query_original,
    qrels,
    eval_metrics=["map", "recall_1000"],
    names=["BM25", "BM25_bo1","BM25_bo2","BM25_klc"]
)
print(out)

print("Q2E/ZS")
data_query_original["query"]=new_queries_keywords
out = pt.Experiment(
    [BM25, BM25_bo1, BM25_bo2, BM25_klc],
    data_query_original,
    qrels,
    eval_metrics=["map", "recall_1000"],
    names=["BM25", "BM25_bo1","BM25_bo2","BM25_klc"]
)
print(out)

print("CoT")
data_query_original["query"]=new_queries_cot
out = pt.Experiment(
    [BM25, BM25_bo1, BM25_bo2, BM25_klc],
    data_query_original,
    qrels,
    eval_metrics=["map", "recall_1000"],
    names=["BM25", "BM25_bo1","BM25_bo2","BM25_klc"]
)
print(out)

