'''
Define constants that are used across multiple files, so they all stay consistent
'''

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
EOS_TOKEN_ID = 128009 # derived from tokenizer.eos_token_id
PAD_TOKEN_ID = EOS_TOKEN_ID # llama has no pad token, so just use EOS
BATCH_SIZE = 3 # there's a good argument this should just live in engine.py, but leave for now
BLOCK_SIZE = 10

# for scheduling choices
PREFILL_ONLY_CUTOFF = 2
DECODE_ONLY_CUTOFF = 4
TIMEOUT = 0
