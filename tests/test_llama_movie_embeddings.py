import unittest
from wsprec.llama_movie_embeddings import build_prompt,tokenize_preserving_prompt,SUFFIX


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=True):return [ord(c) for c in text]
    def decode(self,ids,skip_special_tokens=True):return ''.join(map(chr,ids))


class LlamaPromptTests(unittest.TestCase):
    def test_missing_description_is_exact_title_prompt(self):
        self.assertEqual(build_prompt('Title (1995)',None),build_prompt('Title (1995)','  '))
        self.assertNotIn('Description:',build_prompt('Title (1995)'))

    def test_truncation_preserves_instruction_title_and_readout_suffix(self):
        tok=CharacterTokenizer()
        ids,info=tokenize_preserving_prompt(tok,'Toy Story (1995)','long description '*100,320)
        text=tok.decode(ids)
        self.assertTrue(info['description_truncated'])
        self.assertLessEqual(len(ids),320)
        self.assertTrue(text.endswith(SUFFIX))
        self.assertIn('Title: Toy Story (1995)',text)
        self.assertTrue(text.startswith('Represent the following movie'))

    def test_oversized_title_is_rejected_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            tokenize_preserving_prompt(CharacterTokenizer(),'a'*500,None,100)


if __name__=='__main__':unittest.main()
