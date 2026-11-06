"""Check representation artifacts and content neighbours, never ratings/test."""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'runs/followup/llama3'


def main():
    movies=json.loads((ROOT/'data/movie_descriptions/matched_movies.json').read_text())
    metadata=json.loads((FOLDER/'metadata.json').read_text())
    assert hashlib.sha256((ROOT/'data/movie_descriptions/matched_movies.json').read_bytes()).hexdigest()==metadata['movies_sha256']
    availability=np.load(FOLDER/'description_available.npy')
    arrays={name:np.load(FOLDER/f'{name}.npy') for name in ('title_only','title_description')}
    hashes=json.loads((FOLDER/'prompt_token_hashes.json').read_text())
    assert len(movies)==1682 and availability.shape==(1682,)
    assert int(availability.sum())==metadata['descriptions_available']
    for name,array in arrays.items():
        assert hashlib.sha256((FOLDER/f'{name}.npy').read_bytes()).hexdigest()==metadata['conditions'][name]['embeddings_sha256']
        assert array.shape==(1682,4096)
        assert np.isfinite(array).all()
        assert np.allclose(np.linalg.norm(array,axis=1),1.,atol=2e-6)
    assert np.array_equal(arrays['title_only'][~availability],arrays['title_description'][~availability])
    assert all(hashes['title_only'][i]==hashes['title_description'][i] for i in np.flatnonzero(~availability))
    rng=np.random.default_rng(20261106)
    left=rng.integers(0,len(movies),1000);right=(left+rng.integers(1,len(movies),1000))%len(movies)
    reports={}
    queries=[0,49,126,171,257,299]  # Fixed catalogue indices, not relevance labels.
    for condition,array in arrays.items():
        similarity=array@array.T
        np.fill_diagonal(similarity,-np.inf)
        order=np.argsort(-similarity,axis=1,kind='stable')[:,:10]
        reports[condition]={'mean_random_pair_cosine':float((array[left]*array[right]).sum(1).mean()),
          'fixed_query_examples':[{'item_id':i,'title':movies[i]['title'],'description_available':bool(availability[i]),
             'neighbors':[{'item_id':int(j),'title':movies[j]['title'],'cosine':float(similarity[i,j])} for j in order[i,:5]]} for i in queries],
          'top10_indices':order}
    title_neighbors=reports['title_only'].pop('top10_indices')
    description_neighbors=reports['title_description'].pop('top10_indices')
    counts=np.asarray(json.loads((ROOT/'runs/study/bundles/bpr/common.json').read_text())['item_train_counts'])
    coverage={}
    for name,mask in (('zero',counts==0),('low',(counts>0)&(counts<=8)),('mid',(counts>8)&(counts<=43)),('high',counts>43)):
        coverage[name]={'items':int(mask.sum()),'descriptions_available':int(availability[mask].sum()),'coverage':float(availability[mask].mean())}
    summary={'finite_and_normalized':True,'input_and_embedding_hashes_verified':True,'shape_each':[1682,4096],'exact_missing_fallback_verified':True,
       'missing_items':int((~availability).sum()),'description_coverage_by_train_support':coverage,
       'same_item_title_description_cosine_mean_available':float((arrays['title_only'][availability]*arrays['title_description'][availability]).sum(1).mean()),
       'title_vs_description_top10_overlap_mean_available':float(np.mean([len(set(title_neighbors[i])&set(description_neighbors[i]))/10 for i in np.flatnonzero(availability)])),
       'conditions':reports,'interpretation':'Content-neighbour diagnostics only, not a recommendation performance or semantic benchmark; prompted base Llama embeddings are not guaranteed to be good similarity embeddings.'}
    target=ROOT/'reports/followup/llama3_quality_checks.json'
    target.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    (ROOT/'reports/followup/llama3_embedding_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='conditions'},indent=2))


if __name__=='__main__':main()
