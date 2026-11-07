import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from wsprec import mit_wsp_harness as h


class LiteralEditTests(unittest.TestCase):
    def test_matrix_orientation_and_slice_broadcast(self):
        s=h.zero_state(3,2)
        r,stats=h.apply_edits(s,'```edits\nW1[:,0] = [1,2,3]\nW2[0,0]=1\nW3[0,0]=2\n```')
        self.assertEqual(h.forward(r,np.array([[1,1,1]],np.float32))[0],12)
        self.assertEqual(stats['changed_parameters'],5)
        self.assertTrue(np.array_equal(s['W1'],np.zeros((3,2))))
        self.assertIsNone(stats['relative_delta_l2'])
    def test_fold_uses_current_tensors_and_preserves_shapes(self):
        s=h.zero_state(3,2);s['W1'][:]=[[1,2],[3,4],[5,6]];s['b1'][:]=[1,2]
        r,stats=h.apply_edits(s,'b1[:] += W1[2,:]\nW1[2,:] = 0')
        np.testing.assert_array_equal(r['b1'],[6,8]);np.testing.assert_array_equal(r['W1'][2],[0,0])
        self.assertEqual(r['W1'].shape,(3,2))
    def test_stepped_negative_indices_and_literal_arithmetic(self):
        r,_=h.apply_edits(h.zero_state(4,2),'W1[0:4:2,-1]=[-2,4]\nb1[:] = [1,2]\nb1[:] *= 2\nb3[0] = -(2+3)/2')
        np.testing.assert_array_equal(r['W1'][:,1],[-2,0,4,0]);self.assertEqual(float(r['b3'][0]),-2.5)
    def test_code_execution_and_new_variables_are_rejected(self):
        bad=["import os","W1[:] = open('file')",'for i in range(2):\n W1[i,0]=1',
             'new = 3','W1[:] = W2.T','print(1)','W1 = __import__("os")','W1[:] = [True,False]']
        for text in bad:
            with self.subTest(text=text),self.assertRaises(ValueError):h.apply_edits(h.zero_state(),text)
    def test_bad_shapes_nonfinite_and_bound_rejected(self):
        for text in ('b1[:] = [1,2]','b1[:] = 1e309','b1[:] = 1e7','W1[1000,0]=1','b1[:] = 1/0'):
            with self.subTest(text=text),self.assertRaises(ValueError):h.apply_edits(h.zero_state(),text)
    def test_multiple_edit_blocks_are_ambiguous(self):
        with self.assertRaises(ValueError):h.extract_edits('```edits\nb1[:]=1\n```\n```edits\nb1[:]=2\n```')
    def test_event_audit_fails_closed(self):
        events=[{'type':'thread.started','thread_id':'x'},{'type':'turn.started'},
                {'type':'item.completed','item':{'type':'agent_message','text':'READY'}},
                {'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':1}}]
        a=h.audit_events(events);self.assertTrue(a['tool_free']);self.assertFalse(a['turn_failed']);self.assertEqual(a['usage']['input_tokens'],10)
        self.assertFalse(h.audit_events(events+[{'type':'item.completed','item':{'type':'command_execution'}}])['tool_free'])
        self.assertFalse(h.audit_events(events+[{'type':'new_event'}])['tool_free'])
        self.assertTrue(h.audit_events(events+[{'type':'turn.failed'}])['turn_failed'])
    def test_recovered_transport_notices_are_not_terminal(self):
        events=[{'type':'turn.started'}, {'type':'error','message':'Reconnecting... 2/5 (workspace routing discovery failed)'},
          {'type':'item.completed','item':{'type':'error','message':'Falling back from WebSockets to HTTPS transport. tls handshake eof'}},
          {'type':'item.completed','item':{'type':'agent_message','text':'done'}}, {'type':'turn.completed','usage':{'output_tokens':3}}]
        a=h.audit_events(events);self.assertFalse(a['turn_failed']);self.assertTrue(a['turn_completed']);self.assertEqual(len(a['nonterminal_notices']),2)
    def test_terminal_error_remains_failure_even_with_completion(self):
        events=[{'type':'error','message':'fatal authentication failure'},
          {'type':'item.completed','item':{'type':'agent_message','text':'done'}}, {'type':'turn.completed'}]
        self.assertTrue(h.audit_events(events)['turn_failed'])
        self.assertTrue(h.audit_events(events+[{'type':'turn.failed'}])['turn_failed'])
    def test_transport_error_without_completed_turn_is_failure(self):
        events=[{'type':'error','message':'Reconnecting... 5/5 (workspace routing discovery failed)'}]
        self.assertTrue(h.audit_events(events)['turn_failed'])

    def test_cumulative_trajectory_keeps_last_not_best(self):
        replies=['```edits\nb3[0]=1\n```','```edits\nb3[0]+=2\n```','```edits\nb3[0]*=3\n```']
        calls=[]
        def turn(prompt,outdir,*args,**kwargs):
            p=Path(outdir);p.mkdir(parents=True)
            text=replies[len(calls)];calls.append(prompt)
            return text,{'successful':True,'audit':{'thread_id':'abc'}}
        observed=[]
        def feedback(state):observed.append(float(state['b3'][0]));return 'practice'
        with tempfile.TemporaryDirectory() as tmp,patch.object(h,'run_turn',side_effect=turn):
            result=h.run_trajectory('initial',h.zero_state(),tmp,feedback,2)
            with np.load(Path(tmp)/'final_state.npz') as z:self.assertEqual(float(z['b3'][0]),9)
        self.assertEqual(observed,[1,3]);self.assertEqual(result['forward_batches_used'],2)


if __name__=='__main__':unittest.main()
