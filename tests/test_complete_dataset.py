import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wsprec.complete_dataset import FEATURES, build_records, prepare


class CompleteDatasetTest(unittest.TestCase):
    def setUp(self):
        self.movies = [
            {"item_id": i, "title": str(i), "description": description, "description_available": available}
            for i, (description, available) in enumerate([
                (" first ", True), ("second", True), ("third", True),
                ("   ", True), ("unconfirmed", False), ("unrated", True),
            ])
        ]
        self.raw = np.array([
            [1, 3, 4, 20], [2, 2, 5, 20], [1, 4, 5, 2], [1, 1, 2, 10],
            [1, 2, 3, 20], [2, 1, 2, 10], [3, 5, 1, 4], [1, 5, 1, 4],
        ])

    def test_filter_order_and_determinism(self):
        splits, movies, mapping, stats = build_records(self.raw, self.movies)
        other = build_records(self.raw[::-1], self.movies[::-1])
        self.assertEqual((movies, mapping, stats), other[1:])
        for name in splits:
            np.testing.assert_array_equal(splits[name], other[0][name])
        self.assertEqual(stats["counts"]["after"], {"users": 1, "items": 4, "interactions": 3})
        self.assertEqual(stats["removals"]["fewer_than_three_interactions_after_item_filter"]["users"], 2)
        self.assertEqual(set(mapping["item_new_to_old"]), {0, 1, 2, 5})
        self.assertEqual(mapping["user_new_to_old"], [0])
        for old, name, timestamp in zip(range(3), ("train", "valid", "test"), (10, 20, 20)):
            self.assertEqual(splits[name].shape, (1, 4))
            user, item, _, time = splits[name][0]
            self.assertEqual((user, mapping["item_new_to_old"][item], time), (0, old, timestamp))
        for i, movie in enumerate(movies):
            self.assertEqual(movie["item_id"], i)
            self.assertEqual(movie["description"], self.movies[mapping["item_new_to_old"][i]]["description"].strip())
            self.assertEqual(set(movie), {"item_id", "title", "description"})

    def test_eighty_ten_ten_for_each_user(self):
        movies = [{"item_id": i, "title": str(i), "description": "text", "description_available": True}
                  for i in range(20)]
        raw = np.array([[user, i + 1, 3, i // 2] for i in reversed(range(20)) for user in (9, 2)])
        splits, _, mapping, _ = build_records(raw, movies)
        self.assertEqual(set(mapping["user_new_to_old"]), {1, 8})
        for user in range(2):
            ordered_items = []
            for name, expected in (("train", 16), ("valid", 2), ("test", 2)):
                part = splits[name][splits[name][:, 0] == user]
                self.assertEqual(len(part), expected)
                ordered_items.extend(mapping["item_new_to_old"][item] for item in part[:, 1])
            self.assertEqual(ordered_items, list(range(20)))

    def test_written_embeddings_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, metadata, output = root / "u.data", root / "movies.json", root / "out"
            np.savetxt(raw, self.raw, fmt="%d")
            metadata.write_text(json.dumps(self.movies))
            cache = np.arange(18).reshape(6, 3)
            for i, name in enumerate(FEATURES):
                np.save(root / f"{name}.npy", cache + i)
            prepare(raw, metadata, root, output)
            order = json.loads((output / "mapping.json").read_text())["item_new_to_old"]
            for i, name in enumerate(FEATURES):
                np.testing.assert_array_equal(np.load(output / f"{name}.npy"), (cache + i)[order])
            with np.load(output / "splits.npz") as splits:
                for name in ("train", "valid"):
                    np.testing.assert_array_equal(np.load(output / f"{name}.npy"), splits[name])
            with self.assertRaises(FileExistsError):
                prepare(raw, metadata, root, output)


class DescriptionQualityTest(unittest.TestCase):
    def test_missing_placeholders(self):
        from wsprec.description_quality import has_public_description
        for text in (None, '', '  ', 'No overview found.', ' NO OVERVIEW AVAILABLE! ', 'N/A'):
            self.assertFalse(has_public_description(text), repr(text))
        for text in ('A family returns home.', 'Unknown soldiers seek shelter.', 'Coming soon to a small town, a stranger changes lives.'):
            self.assertTrue(has_public_description(text), repr(text))

    def test_flagged_placeholder_is_removed(self):
        from wsprec.complete_dataset import build_records
        movies=[{'item_id':i,'title':str(i),'description':('No overview found.' if i==3 else 'A public plot.'),'description_available':True} for i in range(4)]
        _, movies, mapping, stats=build_records(np.array([[1,i+1,4,i] for i in range(4)]),movies)
        self.assertEqual(len(movies),3)
        self.assertNotIn(3,mapping['item_new_to_old'])
        self.assertEqual(stats['counts']['after']['interactions'],3)
