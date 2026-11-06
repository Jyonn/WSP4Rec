"""Match public movie descriptions without ever reading interaction labels.

Download mt0rm0/movie_descriptors, then run with --source and --source-info.
Ambiguous/unmatched titles stay missing; no generated plot is substituted.
"""
import argparse
import csv
import difflib
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from .description_quality import has_public_description


def normalize_title(value):
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    value = value.replace('&',' and ').replace('+',' and ')
    articles = 'the|a|an|le|la|les|il|lo|gli|el|los|las|der|die|das|ein|eine'
    value = re.sub(r',\s*('+articles+r"|l')$", '', value)
    value = re.sub(r'^('+articles+r')\s+', '', value)
    value = re.sub(r"^l'", '', value)
    return re.sub(r'[^a-z0-9]', '', value)


def title_variants(title):
    title = re.sub(r'\s*\((?:V|TV)\)\s*$', '', title)
    year_match = re.search(r'\((\d{4})\)\s*$', title)
    year = int(year_match[1]) if year_match else None
    name = title[:year_match.start()].strip() if year_match else title.strip()
    variants = {normalize_title(name)}
    # MovieLens often records English title (original-language title).
    outside = re.sub(r'\([^)]*\)', '', name).strip()
    if outside:
        variants.add(normalize_title(outside))
    for alias in re.findall(r'\(([^)]*)\)', name):
        alias = re.sub(r'^(a\.k\.a\.|aka)\s*', '', alias, flags=re.I)
        if len(normalize_title(alias)) >= 3:
            variants.add(normalize_title(alias))
    return {v for v in variants if v}, year


def match_records(items, source_rows, overrides=None):
    overrides = overrides or {}
    index = defaultdict(set)
    year_index = defaultdict(list)
    for row_id, record in enumerate(source_rows):
        if not has_public_description(record.get('overview')):
            continue
        record['source_row'] = row_id
        variants, _ = title_variants(str(record['name']))
        for key in variants:
            index[key].add(row_id)
        year_index[int(record['release_year'])].append(row_id)
    output, unresolved = [], []

    def unique_descriptions(ids):
        # Duplicated source rows with exactly the same description are harmless.
        unique = {}
        for i in sorted(ids):
            unique.setdefault(str(source_rows[i]['overview']).strip(), i)
        return list(unique.values())

    for item in items:
        variants, year = title_variants(item['title'])
        candidates = set().union(*(index[v] for v in variants)) if variants else set()
        exact = unique_descriptions([i for i in candidates if year == int(source_rows[i]['release_year'])])
        nearby = unique_descriptions([i for i in candidates if year is not None and abs(year-int(source_rows[i]['release_year'])) <= 1])
        override = overrides.get(str(item['item_id']))
        if override is not None:
            chosen, method = int(override['source_row']), 'manual_verified'
            if not override.get('evidence'):
                raise ValueError('manual match needs identity evidence')
        elif len(exact) == 1:
            chosen, method = exact[0], 'exact_title_year'
        elif not exact and len(nearby) == 1:
            chosen, method = nearby[0], 'exact_title_year_plusminus1'
        else:
            chosen, method = None, 'ambiguous' if exact or nearby else 'unmatched'
        row = {**item, 'description': None, 'description_available': False, 'match_method': method,
               'source_row': chosen, 'source_name': None, 'source_year': None}
        if chosen is not None:
            src = source_rows[chosen]
            row.update(description=str(src['overview']).strip(), description_available=True,
                source_name=str(src['name']), source_year=int(src['release_year']))
            if override:
                row['manual_identity_evidence'] = override['evidence']
        else:
            proposed = []
            for y in range((year or 0)-1, (year or 0)+2):
                for i in year_index[y]:
                    src_keys, _ = title_variants(str(source_rows[i]['name']))
                    similarity = max((difflib.SequenceMatcher(None,a,b).ratio() for a in variants for b in src_keys),default=0.)
                    proposed.append((similarity, i))
            row['candidate_proposals'] = [{'source_row':i,'name':str(source_rows[i]['name']),
                'year':int(source_rows[i]['release_year']),'similarity':round(sim,4)} for sim,i in sorted(proposed,reverse=True)[:5]]
            unresolved.append(row)
        output.append(row)
    return output, unresolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--items', type=Path, default=Path('data/ml-100k/u.item'))
    parser.add_argument('--source', type=Path, default=Path('data/movie_descriptions/descriptors_data.parquet'))
    parser.add_argument('--source-info', type=Path, default=Path('data/movie_descriptions/source_info.json'))
    parser.add_argument('--overrides', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/movie_descriptions/matched_movies.json'))
    parser.add_argument('--report', type=Path, default=Path('reports/followup/description_coverage.json'))
    args = parser.parse_args()
    import pyarrow.parquet as pq
    source_rows = pq.read_table(args.source).to_pylist()
    # The dataset card calls this column "name"; the actual parquet uses title.
    for row in source_rows:
        row['name'] = row.get('name') or row.get('title')
        if row['name'] is None:
            raise ValueError('description source missing a movie title')
    items = [{'item_id':int(fields[0])-1,'title':fields[1]} for fields in
        (line.split('|') for line in args.items.read_text(encoding='latin-1').splitlines())]
    overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
    records, missing = match_records(items,source_rows,overrides)
    info = json.loads(args.source_info.read_text())
    available = sum(r['description_available'] for r in records)
    coverage = {'items':len(records),'descriptions_available':available,'coverage':available/len(records),
        'missing':len(missing),'match_methods':dict(Counter(r['match_method'] for r in records)),
        'source':'https://huggingface.co/datasets/mt0rm0/movie_descriptors',
        'source_revision':info['sha'],'source_parquet_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'source_rows':len(source_rows),'source_card_license':'cc0-1.0',
        'matching':'normalized title/alias with release year, unique +/-1 year fallback; no fuzzy autoaccept',
        'unavailable_policy':'preserve item and fall back to title-only embedding; mark missing explicitly; no generated descriptions',
        'raw_data_publication':'not included in git; store source and hashes plus matching metadata',
        'items_sha256':hashlib.sha256(args.items.read_bytes()).hexdigest()}
    for path, value in ((args.output,records),(args.output.with_name('unresolved_movies.json'),missing),(args.report,coverage)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    coverage['matched_records_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.report.write_text(json.dumps(coverage,indent=2,ensure_ascii=False))
    csv_path = args.report.with_name('description_matches.csv')
    with csv_path.open('w',newline='') as f:
        keys=['item_id','title','description_available','match_method','source_row','source_name','source_year']
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in records)
    print(json.dumps(coverage,indent=2))


if __name__=='__main__':
    main()
