import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from wsprec import text_study as study
from wsprec.text_model import TextMF


class TextStudyTests(unittest.TestCase):
    def test_equivalent_patch_containers_preserve_bodies_and_count(self):
        payload=[{'id':str(i),'operations':[]} for i in range(4)]
        self.assertIs(study.unpack_patches(payload),payload)
        self.assertIs(study.unpack_patches({'patches':payload}),payload)
        for invalid in ({'patches':payload,'extra':True},{'patches':payload[:3]}):
            with self.assertRaises(ValueError): study.unpack_patches(invalid)

    def test_effective_projection_change_records_shared_amplification(self):
        torch.manual_seed(2)
        model=TextMF(2,4,2,3)
        base={k:v.detach().clone() for k,v in model.state_dict().items()}
        features=np.ones((4,3),dtype=np.float32)
        edited=study.apply(base,{'text_projection':np.ones((3,2),dtype=np.float32)})
        result=study.effective_change_stats(base,edited,features)
        self.assertEqual(result['item_id_delta_l2'],0.)
        self.assertAlmostEqual(result['text_projection_delta_l2'],np.sqrt(6),places=5)
        self.assertAlmostEqual(result['text_item_contribution_delta_l2'],np.sqrt(72),places=5)
        self.assertAlmostEqual(result['effective_item_delta_l2'],np.sqrt(72),places=5)

    def test_strict_validation_fallback_and_stable_tie(self):
        records=[{'id':'equal','error':None,'valid_ndcg@10':.2},
                 {'id':'invalid','error':'invalid_patch','valid_ndcg@10':None}]
        self.assertIsNone(study.selected(records,.2))
        records += [{'id':'first','error':None,'valid_ndcg@10':.3},
                    {'id':'second','error':None,'valid_ndcg@10':.3}]
        self.assertEqual(study.selected(records,.2)['id'],'first')

    def test_portable_features_contains_projected_and_raw_matrices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            np.save(root/'x.npy',np.ones((4,3),dtype=np.float32))
            np.save(root/'e.npy',np.ones((4,7),dtype=np.float32))
            with patch.object(study,'ROOT',root):
                result=study.portable_features({'projected_features':'x.npy','llama_features':'e.npy'})
                self.assertEqual(set(result),{'projected','llama'})
                self.assertEqual(result['projected'].shape,(4,3))
                self.assertEqual(result['llama'].shape,(4,7))
                self.assertIsNone(study.portable_features({'projected_features':None}))

    def test_controls_select_grid_and_fixed_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'controls.json').write_text(json.dumps({'records':[
                {'method':'fixed_simple','valid_ndcg@10':.2},
                {'method':'grid-a0.25-b0','valid_ndcg@10':.3},
                {'method':'grid-a0.5-b0.25','valid_ndcg@10':.2}]}))
            with patch.object(study,'ROOT',root):
                choices=study.control_choices({'controls':'controls.json'},.2)
            self.assertIsNone(choices['fixed_simple'])
            self.assertEqual(choices['tuned_simple_8']['method'],'grid-a0.25-b0')

    def test_portable_program_reproduces_projection_patch_with_explicit_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'run';data=root/'data';data.mkdir()
            directory=out/'editors/case01';directory.mkdir(parents=True)
            snapshot=out/'snapshots';snapshot.mkdir()
            np.save(data/'train.npy',np.array([[0,0,4,1]],dtype=np.int64))
            torch.manual_seed(2)
            model=TextMF(2,4,2,3)
            base={k:v.detach().clone() for k,v in model.state_dict().items()}
            np.save(root/'x.npy',np.ones((4,3),dtype=np.float32));np.save(root/'e.npy',np.ones((4,7),dtype=np.float32))
            delta=np.full((3,2),.01,dtype=np.float32)
            np.save(snapshot/'w.npy',delta)
            candidate={'id':'one','operations':[{'parameter':'text_projection','op':'add','asset':'w.npy'}]}
            record={'id':'one','error':None,'patch':candidate,'snapshot':'run/snapshots'}
            (directory/'portable.py').write_text('import numpy as np\ndef build(state,train,features):\n assert set(features)=={"projected","llama"}\n return {"one":{"text_projection":np.full_like(state["text_projection"],.01)}}\n')
            assignment={'projected_features':'x.npy','llama_features':'e.npy'}
            with patch.object(study,'ROOT',root),patch.object(study,'OUT',out),patch.object(study,'DATA',data):
                result=study.verify_portable('case01',assignment,base,[record])
            self.assertEqual(result['valid_candidates_reproduced'],1)
            self.assertLess(result['max_absolute_parameter_error'],1e-6)


if __name__=='__main__': unittest.main()
