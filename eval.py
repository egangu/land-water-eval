"""SGLang client: calibrated Base likelihood, IT direct likelihood, native reasoning."""
import argparse
import asyncio
import json
import re
from pathlib import Path
import aiohttp
import numpy as np
from common import DATA, load_grid, map_image

BASE = 'Latitude: {lat}° {ns}, Longitude: {lon}° {ew}.\nQuestion: Is this location land or water?\nAnswer:'
DIRECT = ('Latitude: {lat}° {ns}, Longitude: {lon}° {ew}. Classify this location as Land or Water. '
          'Answer with exactly one word: Land or Water.')
REASON = ('Latitude: {lat}° {ns}, Longitude: {lon}° {ew}. Is this location on land or water? '
          'Reason about the geography, then end with exactly "Answer: Land" or "Answer: Water".')
NULL = 'Latitude: Unknown, Longitude: Unknown.\nQuestion: Is this location land or water?\nAnswer:'

def continuation(tokenizer, text, chat, labels):
    if chat:
        text = tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        if '<|think|>' in text or ('<|turn>' in text and not text.endswith('<|channel>thought\n<channel|>')):
            raise ValueError('Unexpected thinking-off chat template')
    bos = [tokenizer.bos_token_id] if not chat and tokenizer.bos_token_id is not None else []
    prefix = bos + tokenizer.encode(text, add_special_tokens=False)
    full = [bos + tokenizer.encode(text + label, add_special_tokens=False) for label in labels]
    if not all(ids[:len(prefix)] == prefix and len(ids) > len(prefix) for ids in full):
        raise ValueError('Candidate changes the tokenized prompt boundary')
    return prefix, [ids[len(prefix):] for ids in full]

def answer(text):
    if '<|channel>thought' in text and '<channel|>' not in text.rsplit('<|channel>thought', 1)[1]:
        return -1
    text = text.rsplit('<channel|>', 1)[-1]
    text = re.sub(r'(?:\s*(?:<turn\|>|<eos>))+$', '', text).strip()
    match = re.search(r'Answer: (Land|Water)\s*$', text)
    return int(match.group(1) == 'Land') if match else -1
async def evaluate(args):
    from transformers import AutoTokenizer
    rows, truth = load_grid(args.csv)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/(mode+'.png')).exists() for mode in args.mode):
        raise FileExistsError('Use a fresh output directory; existing modes are never repeated')
    limit = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=args.concurrency)) as session:
        async def request(payload):
            async with limit, session.post(args.url.rstrip('/')+'/generate', json=payload) as response:
                response.raise_for_status()
                return await response.json()

        async def likelihood(texts, chat, labels):
            pairs = [continuation(tokenizer, text, chat, labels) for text in texts]
            if all(len(s) == 1 for _, suffixes in pairs for s in suffixes):
                results = await request(dict(input_ids=[p for p, _ in pairs], return_logprob=True,
                    token_ids_logprob=[[s[0] for s in ss] for _, ss in pairs], return_text_in_logprobs=False,
                    sampling_params=dict(temperature=0, max_new_tokens=0)))
                if len(results) != len(pairs):
                    raise ValueError('Incomplete SGLang response')
                scores = []
                for result, (_, suffixes) in zip(results, pairs, strict=True):
                    assert result['meta_info']['completion_tokens'] == 0 and not result.get('text') and not result.get('output_ids')
                    table = {int(e[1]):float(e[0]) for e in result['meta_info']['output_token_ids_logprobs'][0]}
                    assert set(table) == {s[0] for s in suffixes}
                    scores.append([table[s[0]] for s in suffixes])
            else:
                async def forced(prefix, suffix):
                    result = await request(dict(input_ids=prefix+suffix, return_logprob=True,
                        logprob_start_len=len(prefix)-1, sampling_params=dict(temperature=0, max_new_tokens=0)))
                    assert result['meta_info']['completion_tokens'] == 0 and not result.get('text') and not result.get('output_ids')
                    entries = result['meta_info']['input_token_logprobs'][-len(suffix):]
                    assert [int(e[1]) for e in entries] == suffix
                    return sum(float(e[0]) for e in entries)
                flat = await asyncio.gather(*(forced(p, s) for p, ss in pairs for s in ss))
                scores = np.asarray(flat).reshape(len(pairs), 2)
            if not np.isfinite(scores).all():
                raise ValueError('Non-finite candidate likelihood')
            return np.asarray(scores)

        async def reasoning(text):
            ids = tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
                                                add_generation_prompt=True, enable_thinking=True)
            result = await request(dict(input_ids=ids, sampling_params=dict(temperature=0,
                max_new_tokens=args.max_tokens, skip_special_tokens=False)))
            return answer(result['text'])

        summary_path = out/'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        for mode in args.mode:
            template = {'base':BASE, 'direct':DIRECT, 'reasoning':REASON}[mode]
            texts = [template.format(lat=abs(a), ns='N' if a>=0 else 'S', lon=abs(b), ew='E' if b>=0 else 'W') for a,b,_ in rows]
            bias = 0.
            if mode == 'reasoning':
                predictions = np.array(await asyncio.gather(*(reasoning(text) for text in texts)))
            else:
                labels = (' Land', ' Water') if mode == 'base' else ('Land', 'Water')
                if mode == 'base':
                    null = (await likelihood([NULL], False, labels))[0]
                    bias = float(null[0]-null[1])
                batches = await asyncio.gather(*(likelihood(texts[i:i+args.batch_size], mode=='direct', labels)
                                                for i in range(0, len(texts), args.batch_size)))
                scores = np.concatenate(batches)
                predictions = (scores[:, 0]-scores[:, 1] >= bias).astype(int)
            correct = int((predictions == truth).sum())
            summary[mode] = dict(model=args.model, correct=correct, total=len(rows), accuracy=correct/len(rows),
                invalid=int((predictions<0).sum()), bias=bias, max_new_tokens=args.max_tokens if mode=='reasoning' else 0)
            map_image(predictions).save(out/(mode+'.png'))
            summary_path.write_text(json.dumps(summary, indent=2)+'\n')
            print(mode, json.dumps(summary[mode]), flush=True)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', nargs='+', choices=['base','direct','reasoning'], required=True)
    parser.add_argument('--model', required=True, help='Tokenizer path/ID matching the SGLang server')
    parser.add_argument('--csv', default=str(DATA))
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--concurrency', type=int, default=1024)
    parser.add_argument('--max-tokens', type=int, default=8192)
    args = parser.parse_args()
    if len(set(args.mode)) != len(args.mode) or min(args.batch_size,args.concurrency,args.max_tokens) < 1:
        parser.error('Modes must be unique and limits positive')
    asyncio.run(evaluate(args))
if __name__ == '__main__':
    main()
