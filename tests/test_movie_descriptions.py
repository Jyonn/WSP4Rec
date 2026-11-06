import unittest
from wsprec.movie_descriptions import normalize_title,title_variants,match_records


class DescriptionTests(unittest.TestCase):
    def test_article_accents_and_aliases(self):
        self.assertEqual(normalize_title('Cité, The'),normalize_title('The Cite'))
        self.assertEqual(normalize_title('Bonheur, Le'),normalize_title('Le Bonheur'))
        self.assertEqual(normalize_title('Jack & Sarah'),normalize_title('Jack and Sarah'))
        keys,year=title_variants('City of Lost Children, The (Cité des enfants perdus, La) (1995)')
        self.assertEqual(year,1995)
        self.assertIn('cityoflostchildren',keys)

    def test_missing_and_ambiguous_never_filled_by_fuzzy(self):
        source=[{'name':'Example','release_year':1995,'overview':'plot one'},
                {'name':'Example','release_year':1995,'overview':'plot two'},
                {'name':'Exact','release_year':1994,'overview':'exact plot'}]
        items=[{'item_id':0,'title':'Example (1995)'},{'item_id':1,'title':'Exampel (1995)'},
               {'item_id':2,'title':'Exact (1995)'}]
        matched,missing=match_records(items,source)
        self.assertEqual(len(missing),2)
        self.assertEqual(matched[0]['match_method'],'ambiguous')
        self.assertFalse(matched[1]['description_available'])
        self.assertEqual(matched[2]['match_method'],'exact_title_year_plusminus1')

    def test_duplicate_source_and_manual_evidence(self):
        source=[{'name':'Title','release_year':1995,'overview':'same plot'}]*2
        items=[{'item_id':0,'title':'Title (1995)'}]
        matched,_=match_records(items,source)
        self.assertTrue(matched[0]['description_available'])
        with self.assertRaises(ValueError):
            match_records(items,source,{'0':{'source_row':0}})


if __name__=='__main__':unittest.main()
