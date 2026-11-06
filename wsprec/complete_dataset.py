"""Build the description-complete MovieLens catalogue and chronological splits."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from .description_quality import has_public_description


SEED = 20261106
SPLITS = ("train", "valid", "test")
FEATURES = ("title_only", "title_description")


def build_records(raw, matched, seed=SEED):
    """Return splits, public movie metadata, private zero-based mappings and counts."""
    rows = np.asarray(raw, dtype=np.int64).copy()
    if rows.ndim != 2 or rows.shape[1] != 4 or np.any(rows[:, :2] < 1):
        raise ValueError("Expected four columns with one-based user and item IDs")
    rows[:, :2] -= 1
    catalogue = {movie["item_id"]: movie for movie in matched}
    if len(catalogue) != len(matched) or min(catalogue, default=0) < 0:
        raise ValueError("Expected unique nonnegative metadata item IDs")
    if set(rows[:, 1]) - catalogue.keys():
        raise ValueError("Raw item IDs are missing from the metadata")

    retained_items = sorted(item for item, movie in catalogue.items()
                            if movie["description_available"] is True
                            and has_public_description(movie["description"]))
    filtered = rows[np.isin(rows[:, 1], retained_items)]
    original_users = np.unique(rows[:, 0])
    users, counts = np.unique(filtered[:, 0], return_counts=True)
    retained_users = users[counts >= 3]
    rng = np.random.default_rng(seed)
    item_order = rng.permutation(retained_items).tolist()
    user_order = rng.permutation(retained_users).tolist()
    user_ids = {old: new for new, old in enumerate(user_order)}
    item_ids = {old: new for new, old in enumerate(item_order)}

    kept = filtered[np.isin(filtered[:, 0], retained_users)]
    kept = kept[np.lexsort((kept[:, 1], kept[:, 3], kept[:, 0]))]
    parts = {name: [] for name in SPLITS}
    boundaries = np.flatnonzero(np.diff(kept[:, 0])) + 1
    for group in np.split(kept, boundaries):
        n = len(group)
        if n == 0:
            continue
        a = min(max(1, 8 * n // 10), n - 2)
        b = min(max(a + 1, 9 * n // 10), n - 1)
        group[:, 0] = user_ids[int(group[0, 0])]
        group[:, 1] = [item_ids[int(i)] for i in group[:, 1]]
        for key, piece in zip(SPLITS, (group[:a], group[a:b], group[b:])):
            parts[key].append(piece)

    splits = {}
    for name, pieces in parts.items():
        # Keep user blocks anonymous even when row order is visible.
        pieces.sort(key=lambda piece: piece[0, 0])
        splits[name] = np.concatenate(pieces) if pieces else np.empty((0, 4), dtype=np.int64)
    movies = [{"item_id": new, "title": catalogue[old]["title"],
               "description": catalogue[old]["description"].strip()}
              for new, old in enumerate(item_order)]
    mappings = {"id_base": "all IDs are zero-based, including original IDs",
                "item_new_to_old": item_order, "user_new_to_old": user_order}
    diagnostics = {
        "counts": {
            "before": {"users": len(original_users), "items": len(catalogue), "interactions": len(rows)},
            "after": {"users": len(user_order), "items": len(item_order), "interactions": len(kept)},
        },
        "removals": {
            "no_reliable_description_match_in_this_source": {
                "items": len(catalogue) - len(movies), "interactions": len(rows) - len(filtered)},
            "fewer_than_three_interactions_after_item_filter": {
                "users": len(original_users) - len(user_order), "interactions": len(filtered) - len(kept)},
        },
        "split_sizes": {key: len(part) for key, part in splits.items()},
    }
    return splits, movies, mappings, diagnostics


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(raw_path, metadata_path, cache, destination, seed=SEED):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {destination}")
    matched = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = np.loadtxt(raw_path, dtype=np.int64, ndmin=2)
    splits, movies, mappings, manifest = build_records(raw, matched, seed)
    source_paths = {"raw_interactions": raw_path, "matched_movies": metadata_path}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        np.savez(staging / "splits.npz", **splits)
        for key in ("train", "valid"):
            np.save(staging / f"{key}.npy", splits[key])
        save_json(staging / "movies.json", movies)
        save_json(staging / "mapping.json", mappings)
        for name in FEATURES:
            path = cache / f"{name}.npy"
            source_paths[name] = path
            features = np.load(path, mmap_mode="r", allow_pickle=False)
            if (features.ndim != 2 or features.shape[0] != len(matched)
                    or set(m["item_id"] for m in matched) != set(range(len(matched)))):
                raise ValueError(f"Cache and original catalogue IDs must align: {name}")
            indices = np.asarray(mappings["item_new_to_old"], dtype=np.int64)
            np.save(staging / f"{name}.npy", features[indices])
        manifest.update({
            "seed": seed,
            "item_rule": "Reliable matched public description; nonempty and not an explicit missing-description placeholder.",
            "user_rule": "Retain users with at least three interactions after filtering items.",
            "split_rule": "Within user sort by timestamp then original item ID. Floor 80% and 90% "
                          "boundaries, adjusted to leave at least one example in each split.",
            "id_rule": "NumPy default_rng(seed) permutes sorted original items, then sorted retained users.",
            "embedding_source": "Existing per-movie caches subset and reordered by anonymous item ID.",
            "sources": {key: {"path": str(path), "sha256": sha256(path)}
                        for key, path in source_paths.items()},
        })
        save_json(staging / "manifest.json", manifest)
        staging.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/ml-100k/u.data"))
    parser.add_argument("--metadata", type=Path, default=Path("data/movie_descriptions/matched_movies.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/followup/llama3"))
    parser.add_argument("--output", type=Path, default=Path("data/ml-100k-description-complete-v3"))
    args = parser.parse_args()
    result = prepare(args.raw, args.metadata, args.cache, args.output)
    print(json.dumps({key: result[key] for key in ("counts", "removals", "split_sizes")}, indent=2))


if __name__ == "__main__":
    main()
