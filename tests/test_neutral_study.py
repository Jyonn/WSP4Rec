import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from wsprec.cli import FM
from wsprec.neutral_study import evaluate_round,final
from wsprec.tensor_edits import compile_patch


class NeutralStudyTests(unittest.TestCase):
    def test_snapshot_and_feedback_do_not_expose_weight_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';key='bpr-none-blind-r0';folder=out/'editors'/key
            folder.mkdir(parents=True)
            base=FM(2,4,2).state_dict();np.save(folder/'add.npy',np.full((4,1),.01,dtype=np.float32))
            candidates=[{'id':str(i),'operations':[{'parameter':'item_bias.weight','op':'add','asset':'add.npy'}]} for i in range(4)]
            (folder/'round1.json').write_text(json.dumps(candidates))
            train=np.array([[0,0,4,0],[1,1,4,0]]);valid=np.array([[0,1,5,1],[1,2,5,1]])
            with patch('wsprec.neutral_study.OUT',out),patch('wsprec.neutral_study.ROOT',root),patch('wsprec.neutral_study.cp',return_value={'state':base}),patch('wsprec.neutral_study.data',return_value=(train,valid)):
                capture=io.StringIO()
                with contextlib.redirect_stdout(capture):result=evaluate_round(key,1)
                feedback=json.loads(capture.getvalue())['feedback']
                self.assertEqual(set(feedback[0]),{'id','valid_ndcg@10','error'})
                self.assertEqual(sum(r['error']=='invalid_patch' for r in result['results']),3)
                snapshot=root/result['results'][0]['snapshot']
                np.save(folder/'add.npy',np.full((4,1),.2,dtype=np.float32))
                saved,_=compile_patch(base,candidates[0],snapshot)
                np.testing.assert_allclose(saved['item_bias.weight'].numpy(),.01)
                with self.assertRaises(FileExistsError):evaluate_round(key,1)

    def test_malformed_batch_consumes_four_invalid_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';key='mse-none-visible-r0';folder=out/'editors'/key
            folder.mkdir(parents=True);(folder/'round1.json').write_text('[]')
            base=FM(1,2,2).state_dict();rows=np.array([[0,0,4,0]])
            with patch('wsprec.neutral_study.OUT',out),patch('wsprec.neutral_study.ROOT',root),patch('wsprec.neutral_study.cp',return_value={'state':base}),patch('wsprec.neutral_study.data',return_value=(rows,rows)),contextlib.redirect_stdout(io.StringIO()):
                result=evaluate_round(key,1)
            self.assertEqual(len(result['results']),4)
            self.assertTrue(all(r['error']=='invalid_patch' for r in result['results']))

    def test_malformed_operations_do_not_abort_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';key='mse-none-visible-r0';folder=out/'editors'/key
            folder.mkdir(parents=True)
            candidates=[{'id':str(i),'operations':value} for i,value in enumerate((['x'],None,{},4))]
            (folder/'round1.json').write_text(json.dumps(candidates))
            base=FM(1,2,2).state_dict();rows=np.array([[0,0,4,0]])
            with patch('wsprec.neutral_study.OUT',out),patch('wsprec.neutral_study.ROOT',root),patch('wsprec.neutral_study.cp',return_value={'state':base}),patch('wsprec.neutral_study.data',return_value=(rows,rows)),contextlib.redirect_stdout(io.StringIO()):
                result=evaluate_round(key,1)
            self.assertEqual(len(result['results']),4)
            self.assertTrue(all(r['error']=='invalid_patch' for r in result['results']))
            self.assertTrue((out/'evaluations'/key/'round1.json').is_file())

    def test_test_lock_blocks_reopening_before_data_access(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'frozen_manifest.json').write_text(json.dumps({'sha256':{}}))
            (out/'test_opened.lock').write_text('claimed')
            with patch('wsprec.neutral_study.OUT',out),patch('wsprec.neutral_study.data',side_effect=AssertionError('must not access labels')):
                with self.assertRaises(FileExistsError):final()


if __name__=='__main__':unittest.main()
