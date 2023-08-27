# Merge lmd prompts to one cache

import json
import os

prompt_types = ['lmd_negation', 'lmd_numeracy', 'lmd_attribution', 'lmd_spatial']
template_version = "v5.2"
model = "gpt-3.5-turbo"

caches = {}

for prompt_type in prompt_types:
    cache_path = f'cache/cache_{prompt_type.replace("lmd_", "")}{"_" + template_version if template_version != "v5" else ""}_{model}.json'
    with open(os.path.join('..', cache_path), 'r') as f:
        cache = json.load(f)
    caches.update(cache)
    
    print(f"Load path: {cache_path}")
    print(f"Load keys: {len(cache.keys())}")
    print(f"Load keys-value pairs: {len(sum(list(cache.values()), []))}")

prompt_type = 'lmd'
cache_save_path = f'cache/cache_{prompt_type.replace("lmd_", "")}{"_" + template_version if template_version != "v5" else ""}_{model}.json'

print(f"Merged path: {cache_save_path}")
print(f"Merged keys: {len(caches.keys())}")
print(f"Merged keys-value pairs: {len(sum(list(caches.values()), []))}")

with open(os.path.join('..', cache_save_path), 'w') as f:
    json.dump(caches, f, indent=4)
