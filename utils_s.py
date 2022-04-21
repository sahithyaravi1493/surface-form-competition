import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
import sys
# import openai
import time
import os
import re
from statistics import mean, stdev
from abstract_examples import construct_abstractions

def detokenizer(string):
    # ari custom
    string = string.replace("`` ", '"')
    string = string.replace(" ''", '"')
    string = string.replace("` ", '"')
    string = string.replace(" ' ", '" ')
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" :", ":")
    string = string.replace(" ;", ";")
    string = string.replace(" .", ".")
    string = string.replace(" !", "!")
    string = string.replace(" ?", "?")
    string = string.replace(" ,", ",")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    # string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    # string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    # ari custom
    string = string.replace(" n't ", "n't ")
    string = string.replace(" 'd ", "'d ")
    string = string.replace(" 'm ", "'m ")
    string = string.replace(" 're ", "'re ")
    string = string.replace(" 've ", "'ve ")
    return string


def get_key(source, target):
    return '{}'.format(json.dumps({'source':source, 'target':target}))


def gpt3(prompt, max_len, model_name, temp=0, num_log_probs=100, echo=False, n=None):
    print('calling API')
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name, 
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo,
                                                stop='\n',
                                                n=n)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: 
                # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            print("API error:", error)
            time.sleep(1)
    return response

def cross_entropy_list_gpt3(inputs, targets, model_name, batch=None,cache=None, calculate = False):
    '''
    get a list of -log P(target|inp) for
    the inputs and targets in inputs, targets
    using gpt3
    '''
    assert(len(inputs) == len(targets))
    
    ### This block at the top handles caching/batching
    ## basically, first log all computations not in the cache
    ## if calculate is False, return dummy values (just
    ## logging computations to do later)
    ## if calculate is True, do all computations that are not done
    ## then return results for this batch
    ###############################
    ## if we are caching results (LAZY EVALUATION)
    # this is useful for efficient batching. First, add all needed
    # calculations to the batch with calculate = False
    # then run with calculate=True to work through all cached calculations
    if cache is not None:
        # log calculations we have not done yet
        for inp,targ in zip(inputs, targets):
            if get_key(inp, targ) not in cache:
                cache[get_key(inp, targ)] = {'source': inp, 'target':targ,'result':None}
        
        # if not calculating, return dummy values
        if not calculate:
            return [1.]*len(inputs), [1.]*len(inputs), None
        
        # if caching and calculating, we calculate for all examples
        # that have been cached but not calculated
        cache_todo = [(v['source'], v['target']) for v in cache.values() if v['result'] is None]
        
        ## if there are calculations to do, do them
        if len(cache_todo) > 0:
            sources_todo = list(zip(*cache_todo))[0]
            targets_todo = list(zip(*cache_todo))[1]
            
            ce_list, t_len_list, result_list = cross_entropy_list_gpt3(sources_todo, targets_todo,  model_name, cache=None, batch=batch)
            for source, target, ce,t_len, result in zip(sources_todo,targets_todo, ce_list, t_len_list, result_list):
                cache[get_key(source, target)]['ce'] = ce
                cache[get_key(source, target)]['result'] = result
                cache[get_key(source, target)]['t_len'] = t_len
        ## return results for thie example
        output = ([cache[get_key(inp, targ)]['ce'] for inp,targ in zip(inputs, targets)],
                  [cache[get_key(inp, targ)]['t_len'] for inp,targ in zip(inputs, targets)],
                  [cache[get_key(inp, targ)]['result'] for inp,targ in zip(inputs, targets)])
        return output
    ###############################           
    
    
    ### batching ####
    if batch is not None:
        result = {'choices':[]}
        ce_list = []
        len_list = []
        while len(inputs) > 0:
            ce_out, len_out, result_out = cross_entropy_list_gpt3(inputs[:batch], targets[:batch], model_name, cache=None, batch=None)
            inputs, targets = inputs[batch:], targets[batch:]
            
            ce_list = ce_list + ce_out
            len_list = len_list + len_out
            result['choices'] = result['choices'] + result_out
            
            return ce_list, len_list, result['choices']  
    #########
    
    
    #####
    ## calculating cross-entropy
    #####
    data = [inp + targ for inp, targ in zip(inputs, targets)]    
    result = gpt3(data, 0, model_name, echo=True, num_log_probs=1)
    
    #with open(out_file, 'a') as out:
    #    out.write(f'{json.dumps(result)}\n')
    ce_list = []
    t_lens = []
    for inp, out in zip(inputs, result['choices']):
        # get the beginning of the target from the response (based on tokenization)
        i = 0
        while out['logprobs']['text_offset'][i] < len(inp):
            i += 1
        t_lens.append(len(out['logprobs']['text_offset']) - i)
        # sum of log probs over the target tokens
        ce = -sum(out['logprobs']["token_logprobs"][i:])
        ce_list.append(ce)
    return ce_list, t_lens, result['choices'] 


def cross_entropy_list(sources, targets, model, cache = None, batch=False, calculate=True):
    '''
    Gets a list of CE values, where the ith item is a list of cross-entropies
    for targets[i] with sources[i] as contexts

    targets and sources are lists of lists of tokens (integers)

    model is a language model

    batch is the batch size to break things up into, batch=False means don't
    break things up into batches, do them all in one go.
    
    CACHING:
    
    cache is a dictionary for single source/target pairs
      accessed by cache[get_key(source,target)]
      it has fields source, target, result
    
    calculate decides whether to immediates calculate for batch of input
      sources/targets or just log them as todo in the cache. To efficiently 
      batch, we can first log many todo calculations by calling cross_entropy_list
      multiple times with calculate=False and the same input cache
      Then finally calling it with calculate=True which will then catch up on all
      todo calculations, caching them together efficiently
    
    '''
    
    ###############################
    # This block handles caching of results (LAZY EVALUATION)
    # this is useful for efficient batching. First, add all todo
    # calculations to the cache with calculate = False (won't do them yet)
    # then run with calculate=True to work through all cached calculations
    # in efficient batches
    if cache is not None:

        # log calculations we have not done yet
        for source,target in zip(sources, targets):
            if get_key(source, target) not in cache:
                cache[get_key(source, target)] = {'source': source, 'target':target,'result':None, 'worst_tkn': None}
        
        # if not calculating, return dummy values
        if not calculate:
            return [1.]*len(sources), [1.]*len(sources)
        
        # if caching and calculating, we calculate for all examples
        # that have been cached but not calculated
        cache_todo = [(v['source'], v['target']) for v in cache.values() if v['result'] is None]
        
        ## if there are calculations to do, do them
        if len(cache_todo) > 0:
            sources_todo = list(zip(*cache_todo))[0]
            targets_todo = list(zip(*cache_todo))[1]
            
            cache_results, worst_tkn_results = cross_entropy_list(sources_todo, targets_todo, model, cache=None, batch=batch)
            for source, target, result, w in zip(sources_todo,targets_todo, cache_results, worst_tkn_results):
                cache[get_key(source, target)]['result'] = result
                cache[get_key(source, target)]['worst_tkn'] = w

    
        ## return results for thie example
        results = [cache[get_key(source, target)]['result'] for source,target in zip(sources, targets)]
        results_worst_tkn = [cache[get_key(source, target)]['worst_tkn'] for source,target in zip(sources, targets)]
        return results, results_worst_tkn
    ###############################        
        
        
        
        
    
    
    
    assert(len(sources ) == len(targets))
    n_seqs = len(sources)
    
    # torch.cuda.empty_cache()
    device = model.transformer.wte.weight.device

    # if batching, break it up into smaller pieces
    if batch:
        ce_list = []
        wt_list = []
        
        n_batches = math.ceil(len(sources) / batch)
        
        list_fun = (lambda v: tqdm(list(v))) if cache is not None else list
        
        for i in tqdm(list(range(n_batches))):
            x, y = cross_entropy_list(sources[i*batch:(i+1)*batch], targets[i*batch:(i+1)*batch], model, batch=False)
            ce_list += x
            wt_list += y
            #sources, targets = sources[batch:], targets[batch:]
        return ce_list, wt_list

    # initialize input tensors
    max_len = max([len(s + t) for s,t in zip(sources, targets)])
    input_ids = torch.zeros((n_seqs, max_len)).long() 
    #-100 is the padding token, which is ignored by F.cross_entropy below
    labels = -100*torch.ones((n_seqs, max_len)).long()
    
    # for each source, target pair, set values in the input tensors
    for i, (source, target) in enumerate(zip(sources,targets)):
        s = torch.tensor(source).long()
        t = torch.tensor(target).long()
        input_ids[i,:len(s)] = s
        input_ids[i,len(s):len(s) + len(t)] = t
        # ignore all predictions except in the target span
        labels[i,len(s):len(s) + len(t)] = t
    
    # get logits from the model
    with torch.no_grad():
        input_ids = input_ids.to(device)
        logits = model(input_ids, return_dict=True)
        #print(logits)
        logits = logits.logits.cpu()[:,:-1].contiguous()
    
    # get cross-entropies given the logits
    logit_shape = logits.shape
    logits = logits.view(-1, logit_shape[-1])
    ce_list_orig = F.cross_entropy(logits, labels[:,1:].contiguous().view(-1), reduction='none') # log softmax + NLL
    
    ce_list_orig = ce_list_orig.view(n_seqs, max_len -1)
    ce_mins, _ = torch.topk(ce_list_orig,2) # take the 2 tokens with max loss
    # print(ce_mins)
    ce_mins = torch.sum(ce_mins, dim=1) # sum the losses of the tokens
    # print("summed", ce_mins)
    ce_mins = ce_mins.squeeze().tolist()
    ce_list = ce_list_orig.sum(dim=1).squeeze().tolist()

    
    # if one element (i.e. len(sources) == 1), nest it into a list. Otherwise, give full list
    # this just handles an idiosyncracy of the .tolist() function
    try:
        len(ce_list)
    except:
        ce_list = [ce_list]
        ce_mins = [ce_mins]
    
    return ce_list, ce_mins





def inference_autobatch( model, encoder, example, batch = 1, prelog = False, cache = None):
    '''
    
    if prelog is true, then we're just logging calculations to do in one big batch calculate
    (used for caching)
    
    
    '''
    
    ## if we are just prelogging cross entropy calculations to do later,
    ## we will set caclulate=False for cross_entropy_list and it will output
    ## a dummy value for now and just log calculations to do. Then the output
    ## of inference_autobatch will not be correct, calling it in this case is 
    ## just to log calculations to do in big batches
    if prelog and (cache is not None):
        calculate = False 
    else:
        calculate = True
    
    
    #####
    ## input data handling
    #####
    # i.e. if we're using GPT-3 through the OpenAI API
    if type(model) == str:
        max_len = 2048  
        gpt3 = True
    else:
        max_len = 1024
        gpt3 = False

    options = []
    for opt_raw in example['options']:
        if gpt3:
            options.append(opt_raw)
        else:
            # first, encode the option 
            opt = { key: encoder.encode(opt_raw[key]) for key in opt_raw.keys() }

            ## trim the option to the max length for gpt2
            opt['premise'] = opt['premise'][-(max_len - len(opt['hypothesis'])):]
            assert(len(opt['premise'] + opt['hypothesis']) <= max_len)

            # then add the encoded, trimmed option
            options.append( opt )

    #####
    ## cross-entropy calculation
    #####
    if gpt3:
        ## get conditional CEs
        cond_ce, cond_t_lens, _ = cross_entropy_list_gpt3([opt['premise'] for opt in options], 
                                                          [opt['hypothesis'] for opt in options],
                                                          model,
                                                        cache=cache,calculate = calculate, batch=batch)
        
        ## get domain conditional CEs
        domain_cond_ce, domain_cond_t_lens, _ = cross_entropy_list_gpt3([opt['uncond_premise'] for opt in options],
                                        [opt['uncond_hypothesis'] for opt in options],
                                        model,
                                        cache=cache,calculate = calculate, batch=batch)

        ## get unconditional CEs
        uncond_ce, uncond_t_lens, _ = cross_entropy_list_gpt3([':' for opt in options],
                                        [opt['uncond_hypothesis'] for opt in options],
                                        model,
                                        cache=cache,calculate = calculate, batch=batch)
    else:
        ## get conditional CEs
        cond_ce, wt = cross_entropy_list([opt['premise'] for opt in options], 
                                    [opt['hypothesis'] for opt in options],
                                    model, cache=cache, batch=batch, calculate = calculate)

        
        ## get domain conditional CEs
        domain_cond_ce, _ = cross_entropy_list([opt['uncond_premise'] for opt in options],
                                        [opt['uncond_hypothesis'] for opt in options],
                                        model, cache=cache, batch=batch, calculate = calculate)
        
        ## get unconditional CEs
        uncond_ce , _ = cross_entropy_list([[25] for opt in options],
                                       [opt['uncond_hypothesis'] for opt in options],
                                       model, cache=cache, batch=batch, calculate = calculate)

    ## get average CE by token
    if gpt3:
        avg_cond_ce = [ce/l for ce, l in zip(cond_ce, cond_t_lens)]
    else:
        
        avg_cond_ce = [ce / len(opt['hypothesis']) for ce, opt in zip(cond_ce, options)]
       
    
    #####
    ## prediction
    #####
    # calculate dcpmi
    dcpmi = [ce_0 - ce_1 for ce_0,ce_1 in zip(domain_cond_ce, cond_ce)]
    pmi = [ce_0 - ce_1 for ce_0,ce_1 in zip(uncond_ce, cond_ce)]

    
    ## make predictions based on different scores
    lm_pred = cond_ce.index(min(cond_ce))
    lm_pred_wt = wt.index(min(wt))
    lm_avg_pred = avg_cond_ce.index(min(avg_cond_ce))
    lm_domain_cond_pred = domain_cond_ce.index(min(domain_cond_ce))
    dcpmi_pred = dcpmi.index(max(dcpmi))
    pmi_pred = pmi.index(max(pmi))
    pred = {
                 'lm': lm_pred,
                 'lm_wt': lm_pred_wt,
                 'tok_mean': lm_avg_pred,
                 'dcpmi' : dcpmi_pred,
                 'pmi': pmi_pred,
                 'domain_cond': lm_domain_cond_pred,
           }
    return pred

hypernym_cache = {}
synonym_cache = {}
def process_abstractions(opt, opt_raw, encoder, mode):
    # Get abstractions of premise
    opt['raw_premise'] = opt_raw['premise']
    opt['raw_hypothesis'] = opt_raw['hypothesis']
    if mode == 'premise':
        N_P = 5
        N_H = 1
        if opt_raw['premise'] in hypernym_cache:
             premise_hypernyms, premise_synonyms = hypernym_cache[opt_raw['premise']], synonym_cache[opt_raw['premise']]
        else:
            premise_hypernyms, premise_synonyms = construct_abstractions(opt_raw['premise'])
            premise_hypernyms = premise_hypernyms[:N_P] + [opt_raw['premise']]
            premise_synonyms = premise_synonyms[:N_P] + [opt_raw['premise']]
        hypothesis_hypernyms, hypothesis_synonyms = [opt_raw['hypothesis']], [opt_raw['hypothesis']]
    elif mode == 'hypothesis':
        N_P = 1
        N_H = 5
        if opt_raw['hypothesis'] in hypernym_cache:
             hypothesis_hypernyms, hypothesis_synonyms = hypernym_cache[opt_raw['hypothesis']], synonym_cache[opt_raw['hypothesis']]
        else:
            # Get abstractions of hypothesis
            hypothesis_hypernyms, hypothesis_synonyms = construct_abstractions(opt_raw['hypothesis'])
            hypothesis_hypernyms = hypothesis_hypernyms[:N_H] + [opt_raw['hypothesis']]
            hypothesis_synonyms = hypothesis_synonyms[:N_H] + [opt_raw['hypothesis']]
        premise_hypernyms, premise_synonyms = [opt_raw['premise']], [opt_raw['premise']]
    else:
        N_P = 5
        N_H = 5
        # Get abstractions of premise
        if opt_raw['premise'] in hypernym_cache:
             premise_hypernyms, premise_synonyms = hypernym_cache[opt_raw['premise']], synonym_cache[opt_raw['premise']]
        else:
            premise_hypernyms, premise_synonyms = construct_abstractions(opt_raw['premise'])
            premise_hypernyms = premise_hypernyms[:N_P] + [opt_raw['premise']]
            premise_synonyms = premise_synonyms[:N_P] + [opt_raw['premise']]
        hypothesis_hypernyms, hypothesis_synonyms = [opt['hypothesis']], [opt['hypothesis']]
        # Get abstractions of hypothesis
        if opt_raw['hypothesis'] in hypernym_cache:
             hypothesis_hypernyms, hypothesis_synonyms = hypernym_cache[opt_raw['hypothesis']], synonym_cache[opt_raw['hypothesis']]
        else:
            # Get abstractions of hypothesis
            hypothesis_hypernyms, hypothesis_synonyms = construct_abstractions(opt_raw['hypothesis'])
            hypothesis_hypernyms = hypothesis_hypernyms[:N_H] + [opt_raw['hypothesis']]
            hypothesis_synonyms = hypothesis_synonyms[:N_H] + [opt_raw['hypothesis']]

    # cache already seen abstractions saves time!!
    hypernym_cache[opt_raw['premise']] = premise_hypernyms
    synonym_cache[opt_raw['premise']] = premise_synonyms
    hypernym_cache[opt_raw['hypothesis']] = hypothesis_hypernyms
    synonym_cache[opt_raw['hypothesis']] = hypothesis_synonyms

    opt['hypernym_premise'] = [encoder.encode(a) for a in premise_hypernyms]
    opt['synonym_premise'] = [encoder.encode(a) for a in premise_synonyms]
    opt['hypernym_hypothesis'] = [encoder.encode(a) for a in hypothesis_hypernyms]
    opt['synonym_hypothesis'] = [encoder.encode(a) for a in hypothesis_synonyms]
    opt['raw_synonym_hypothesis'] = hypothesis_synonyms
    opt['raw_hypernym_hypothesis'] = hypothesis_hypernyms
    opt['raw_synonym_premise'] = premise_synonyms
    opt['raw_hypernym_premise'] = premise_hypernyms
    

    return opt, N_H, N_P


def inference_autobatch_abstracted( model, encoder, example, batch = 1, prelog = False, cache = None, mode='premise', stem=''):
    '''
    
    if prelog is true, then we're just logging calculations to do in one big batch calculate
    (used for caching)
    
    
    '''
    
    ## if we are just prelogging cross entropy calculations to do later,
    ## we will set caclulate=False for cross_entropy_list and it will output
    ## a dummy value for now and just log calculations to do. Then the output
    ## of inference_autobatch will not be correct, calling it in this case is 
    ## just to log calculations to do in big batches
    if prelog and (cache is not None):
        calculate = False 
    else:
        calculate = True
    
    #####
    ## input data handling
    #####
    # i.e. if we're using GPT-3 through the OpenAI API
    if type(model) == str:
        max_len = 2048  
        gpt3 = True
    else:
        max_len = 1024
        gpt3 = False

    options = []
    # print(example)
    for opt_raw in example['options']:
        if gpt3:
            options.append(opt_raw)
        else:
            # first, encode the option 
            opt = { key: encoder.encode(opt_raw[key]) for key in opt_raw.keys() }
            ## trim the option to the max length for gpt2
            opt['premise'] = opt['premise'][-(max_len - len(opt['hypothesis'])):]
            assert(len(opt['premise'] + opt['hypothesis']) <= max_len)
            opt, N_H, N_P = process_abstractions(opt, opt_raw, encoder, mode)
            options.append(opt)

    #####
    ## cross-entropy calculation
    #####
    if gpt3:
        ## get conditional CEs
        cond_ce, cond_t_lens, _ = cross_entropy_list_gpt3([opt['premise'] for opt in options], 
                                                          [opt['hypothesis'] for opt in options],
                                                          model,
                                                        cache=cache,calculate = calculate, batch=batch)
        
        ## get domain conditional CEs
        domain_cond_ce, domain_cond_t_lens, _ = cross_entropy_list_gpt3([opt['uncond_premise'] for opt in options],
                                        [opt['uncond_hypothesis'] for opt in options],
                                        model,
                                        cache=cache,calculate = calculate, batch=batch)

        ## get unconditional CEs
        uncond_ce, uncond_t_lens, _ = cross_entropy_list_gpt3([':' for opt in options],
                                        [opt['uncond_hypothesis'] for opt in options],
                                        model,
                                        cache=cache,calculate = calculate, batch=batch)
    else:
        ## get conditional CEs
        cond_ce, wt = cross_entropy_list([opt['premise'] for opt in options], 
                                    [opt['hypothesis'] for opt in options],
                                    model, cache=cache, batch=batch, calculate = calculate)
        # print("OPTIONS", len(options))
        # print("PREMISE RAW", [opt['raw_premise'] for opt in options])
        # print("HYPOTHESIS RAW", [opt['raw_hypothesis'] for opt in options])

        ## get conditional CEs for all hypernyms
        all_ces = [cond_ce]
        all_wts = [wt]
        raw_text = []
        for i in range(N_P):
            for j in range(N_H):
                abs_cond_ce, abs_wt = cross_entropy_list( 
                                            [opt['hypernym_premise'][i] for opt in options], 
                                            [opt['hypernym_hypothesis'][j] for opt in options],
                                            model, cache=cache, batch=batch, calculate = calculate)
                raw_text.append([(opt['raw_hypernym_premise'][i] + '<BREAK>' + opt['raw_hypernym_hypothesis'][j]) for opt in options])
      
                all_ces.append(abs_cond_ce)
                all_wts.append(abs_wt)



        # print([[all_ces[j], raw_text[j] for j in range(len(all_ces))] for i in range(len(options))])
        abstrated_ce = [min([item[i] for item in all_ces]) for i in range(len(options))]
        abstracted_avg = [mean([item[i] for item in all_ces]) for i in range(len(options))]
        abstracted_sd = [stdev([item[i] for item in all_ces]) for i in range(len(options))]

        # Print scores of hypernym abstractions
        if not prelog:
            for i in range(len(options)): # each array has a tiny array inside of length = options
                for item1, item2, item3 in zip(raw_text, all_ces, all_wts):
                    with open(f"{stem}/scores_hyp.txt", "a") as myfile:
                        myfile.write(f",{item1[i]},{item2[i]},{item3[i]}\n")

            with open(f"{stem}/sd_hyp.txt", "a") as myfile:
                myfile.write(str(abstracted_sd[0]) + "\n")

        abstrated_wt_ce = [min([item[i] for item in all_wts]) for i in range(len(options))]
        # print("Hypernym most plausible ", abstrated_ce)

        ## get conditional CEs for all synonyms
        all_syn_ces = [cond_ce]
        all_syn_wts = [wt]
        raw_text_s = []
        for i in range(N_P):
            for j in range(N_H):
                syn_cond_ce, syn_wt = cross_entropy_list( 
                                                [opt['synonym_premise'][i] for opt in options], 
                                                [opt['synonym_hypothesis'][j] for opt in options],
                                                model, cache=cache, batch=batch, calculate = calculate)
                raw_text_s.append([(opt['raw_synonym_premise'][i] + '<BREAK>' + opt['raw_synonym_hypothesis'][j]) for opt in options])
                all_syn_ces.append(syn_cond_ce)
                all_syn_wts.append(syn_wt)

        syn_ce = [min([item[i] for item in all_syn_ces]) for i in range(len(options))]
        syn_avg = [mean([item[i] for item in all_syn_ces]) for i in range(len(options))]
        syn_sd = [stdev([item[i] for item in all_syn_ces]) for i in range(len(options))]
        syn_wt_ce = [min([item[i] for item in all_syn_wts]) for i in range(len(options))]
        # Print scores of synonym abstractions
        if not prelog:
            for i in range(len(options)): # each array has a tiny array inside of length = options
                for item1, item2, item3 in zip(raw_text_s, all_syn_ces, all_syn_wts):
                    with open(f"{stem}/scores_syn.txt", "a") as myfile:
                        myfile.write(f"{item1[i]},{item2[i]},{item3[i]}\n")

            with open(f"{stem}/sd_syn.txt", "a") as myfile:
                myfile.write(str(syn_sd[0]) + "\n")

        ## get ce by including synonym and hypernym
        all_ces.extend(all_syn_ces)
        all_wts.extend(all_syn_wts)
        both_ce = [min([item[i] for item in all_ces]) for i in range(len(options))]
        both_avg = [mean([item[i] for item in all_ces]) for i in range(len(options))]
        both_wt_ce = [min([item[i] for item in all_wts]) for i in range(len(options))]


        ## get domain conditional CEs
        domain_cond_ce, _ = cross_entropy_list([opt['uncond_premise'] for opt in options],
                                        [opt['uncond_hypothesis'] for opt in options],
                                        model, cache=cache, batch=batch, calculate = calculate)
        
        ## get unconditional CEs
        uncond_ce , _ = cross_entropy_list([[25] for opt in options],
                                       [opt['uncond_hypothesis'] for opt in options],
                                       model, cache=cache, batch=batch, calculate = calculate)

    ## get average CE by token
    if gpt3:
        avg_cond_ce = [ce/l for ce, l in zip(cond_ce, cond_t_lens)]
    else:
        
        avg_cond_ce = [ce / len(opt['hypothesis']) for ce, opt in zip(cond_ce, options)]
       
    
    #####
    ## prediction
    #####
    # calculate dcpmi
    dcpmi = [ce_0 - ce_1 for ce_0,ce_1 in zip(domain_cond_ce, cond_ce)]
    pmi = [ce_0 - ce_1 for ce_0,ce_1 in zip(uncond_ce, cond_ce)]

    
    ## make predictions based on different scores
    lm_pred = cond_ce.index(min(cond_ce))
    lm_pred_wt = wt.index(min(wt))
    lm_abs = abstrated_ce.index(min(abstrated_ce))
    lm_abs_avg = abstracted_avg.index(min(abstracted_avg))
    lm_abs_wt = abstrated_wt_ce.index(min(abstrated_wt_ce))
    lm_syn = syn_ce.index(min(syn_ce))
    lm_syn_avg = syn_avg.index(min(syn_avg))
    lm_syn_wt = syn_wt_ce.index(min(syn_wt_ce))
    lm_syn_hyp = both_ce.index(min(both_ce))
    lm_syn_hyp_avg = both_avg.index(min(both_avg))
    lm_syn_hyp_wt = both_wt_ce.index(min(both_wt_ce))
    lm_avg_pred = avg_cond_ce.index(min(avg_cond_ce))
    lm_domain_cond_pred = domain_cond_ce.index(min(domain_cond_ce))
    dcpmi_pred = dcpmi.index(max(dcpmi))
    pmi_pred = pmi.index(max(pmi))
    pred = {
                 'lm': lm_pred,
                 'lm_wt': lm_pred_wt,
                 'lm_hyp': lm_abs,
                 'lm_hyp_avg': lm_abs_avg,
                 'lm_hyp_wt': lm_abs_wt,
                 'lm_syn': lm_syn,
                 'lm_syn_avg': lm_syn_avg,
                 'lm_syn_wt': lm_syn_wt,
                 'lm_syn_hyp': lm_syn,
                 'lm_syn_hyp_avg': lm_syn_avg,
                 'lm_syn_hyp_wt': lm_syn_wt,
                 'tok_mean': lm_avg_pred,
                 'dcpmi' : dcpmi_pred,
                 'pmi': pmi_pred,
                 'domain_cond': lm_domain_cond_pred,
           }
    return pred

def fwd(model, encoder, examples, batch, cache = None, abstraction_method='none', stem=''):
    '''
    This is designed for gpt2-style language models
    
    Inputs: (any you don't know)
        model - a HuggingFace Transformers gpt-2 model

        encoder - a HuggingFace Transformers tokenizer

        examples = [ex1, ex2, ...]
            where ex = [opt1, opt2, ...] (multiple choice options)
            where opt = (premise, hypothesis) 
        
        batch: is the max allowed batch size (set to 1 for no batching)
    '''
    
    if type(model) != str:
        # print the first example to make sure the format is ok
        print('='*50)
        print('MAKE SURE TOKENIZATION AND FORMATTING LOOKS OK')
        for i in range(0,2):
            print(f'\nprint example {i} of {len(examples)}')
            ex = examples[i]
            options = ex['options']
            opt = options[i]
            print('CONDITIONAL:')
            print(encoder.decode(encoder.encode(opt['premise'])) + '<BREAK>' + encoder.decode(encoder.encode(opt['hypothesis'])))
            print('UNCONDITIONAL:')
            print(encoder.decode(encoder.encode(opt['uncond_premise'])) + '<BREAK>' + encoder.decode(encoder.encode(opt['uncond_hypothesis'])))
            print('='*50)
    else:
        # print the first example to make sure the format is ok
        print('='*50)
        print('MAKE SURE TOKENIZATION AND FORMATTING LOOKS OK')
        print('\nprint example 0 of {}:'.format(len(examples)))
        ex = examples[0]
        options = ex['options']
        opt = options[0]
        print('CONDITIONAL:')
        print(opt['premise'] + '<BREAK>' + opt['hypothesis'])
        print('UNCONDITIONAL:')
        print(opt['uncond_premise'] + '<BREAK>' + opt['uncond_hypothesis'])
        print('='*50)

    predictions_list = []
    

    ## in this loop, prelog is set to true so we are just logging cross_entropy_list calculations
    ## but not doing them yet
    if cache is not None:
        print('logging examples')
        for example in tqdm( examples):
            _ = inference_autobatch_abstracted(model, encoder, example, prelog=True, cache = cache, batch=batch, mode=abstraction_method, stem=stem)

    ## in this loop, we actually do the calculations from above in efficient batches, storing results 
    ## in the cache and calculating actual predictions
    print('actually calculating')
    for example in tqdm(examples):
        pred = inference_autobatch_abstracted(model, encoder, example, prelog=False, cache = cache, batch=batch, mode=abstraction_method, stem=stem)
        predictions_list.append(pred)

    labels = [ex['label'] for ex in examples]
    # get predictions into list by scoring key
    predictions_dict = {key:list(map(lambda v: v[key], predictions_list)) for key in predictions_list[0].keys()}

    # calculate accuracies
    results = {key: sum(list(map(lambda v: v[0] == v[1], zip(predictions_dict[key] , labels) )))/len(labels) for key in predictions_dict.keys()}

    # save labels for later
    predictions_dict['labels'] = labels
    return results, predictions_dict

def score(model, model_name, encoder, examples, stem, split, batch, abstraction_method):
    hist_path = f'{stem}{model_name}-{split}.hist'
    
    if not os.path.exists(hist_path):
        cache = {}
        with open(hist_path, 'w') as f:
            f.write(json.dumps(cache))
    else:
        MB = os.path.getsize(hist_path)/1000000
        print('='*50)
        print('Loading existing cache, size {} MB'.format(MB))
        print('='*50)
        
    with open(hist_path, 'r') as f:
        cache = json.loads(f.read())
        
    accs, preds = fwd(model, encoder, examples, batch, cache, abstraction_method, stem)
    
    print('='*50)
    print('saving cache to {}'.format(hist_path))
    print('='*50)
    with open(hist_path, 'w') as f:
        f.write(json.dumps(cache))

    # save scores
    results_path = f'{stem}{split}.accs'
    with open(results_path,'w') as out:
        out.write(json.dumps(accs))

    # save predicted labels
    preds_path = f'{stem}{split}.preds'
    with open(preds_path, 'w') as out:
        out.write(json.dumps(preds))

    return accs
