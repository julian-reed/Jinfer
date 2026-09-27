'''
Define constants that are used across multiple files, so they all stay consistent
'''

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
EOS_TOKEN_ID = 128009 # derived from tokenizer.eos_token_id
BATCH_SIZE = 1 # there's a good argument this should just live in engine.py, but leave for now
