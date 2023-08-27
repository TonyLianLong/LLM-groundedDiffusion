# If for a prompt we query fewer or equal to the times we have cache, we return from the cache sequentially. Otherwise we store into cache.
# Need to set up a new cache if the hyperparam or the template changes.

import os
import pickle, json

cache_path = ''
cache_format = 'json'

global_cache = {}

# The cache records the access times to load more than one value in the cache when the keys repeat.
global_cache_index = {}
# This is for export and debugging the queries
cache_queries = {}

def reset_cache_access():
    global global_cache_index, cache_queries
    global_cache_index = {}
    cache_queries = {}

def values_accessed():
    return sum(global_cache_index.values())

def init_cache(allow_nonexist=True):
    global global_cache
    assert cache_path, "Need to set cache path"
    
    print(f"Cache path: {cache_path}")
    
    if not allow_nonexist:
        assert os.path.exists(cache_path), f"{cache_path} does not exist"
    
    if os.path.exists(cache_path):
        if cache_format == "pickle":
            with open(cache_path, 'rb') as f:
                global_cache = pickle.load(f)
        elif cache_format == "json":
            with open(cache_path, 'r') as f:
                global_cache = json.load(f)

def get_cache(key):
    if key not in global_cache:
        global_cache[key] = []
        
    if key not in global_cache_index:
        global_cache_index[key] = 0
    
    current_items = global_cache[key]
    current_index = global_cache_index[key]
    if len(current_items) > current_index:
        global_cache_index[key] += 1
        if key not in cache_queries:
            cache_queries[key] = []
        cache_queries[key].append(current_items[current_index])
        return current_items[current_index]
    
    return None
    
def add_cache(key, value):
    global_cache_index[key] += 1
    global_cache[key].append(value)
    
    if cache_format == "pickle":
        with open(cache_path, 'wb') as f:
            pickle.dump(global_cache, f)
    elif cache_format == "json":
        with open(cache_path, 'w') as f:
            json.dump(global_cache, f, indent=4)
    
    return value
    
def pkl_to_json(filename):
    assert 'pkl' in filename, filename
    with open(filename, 'rb') as f:
        cache = pickle.load(f)
    del f
    filename = filename.replace('pkl', 'json')
    assert not os.path.exists(filename)
    print(cache)
    with open(filename, 'w') as f:
        json.dump(cache, f, indent=4)
