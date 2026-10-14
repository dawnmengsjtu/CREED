"""Replay M/GDSC extraction verification without re-reading raw responses."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    c=json.loads((ROOT/'m_gdsc_config.json').read_text());qa=json.loads((ROOT/'m_gdsc_extra19_QA.json').read_text())
    with np.load(ROOT/'m_gdsc_extra19.npz',allow_pickle=False) as z,np.load(ROOT/'m_gdsc_old_train_parity_reference.npz',allow_pickle=False) as r:
        np.testing.assert_array_equal(z['sample_ids'],[x['sample_id'] for x in c['candidates']]);assert not set(z['sample_ids'])&set(c['excluded_old608'])
        for field,col in [('model_ids','model_id'),('patient_ids','patient_id'),('tissues','tissue')]:np.testing.assert_array_equal(z[field],[x[col] for x in c['candidates']])
        np.testing.assert_array_equal(z['m_feature_ids'],c['m_feature_ids']);np.testing.assert_array_equal(z['drug_ids'],c['drug_ids'])
        assert z['m'].shape==(19,225) and z['auc'].shape==(19,295) and np.isfinite(z['m']).all()
        np.testing.assert_array_equal(z['parity_sample_ids'],r['sample_ids'])
        for k in ('m','auc'):np.testing.assert_array_equal(z['parity_'+k],r[k])
        assert int(np.isfinite(z['auc']).sum())==qa['auc_observed']==4485
        assert int(np.isnan(z['auc']).sum())==qa['auc_missing']==1120
        assert qa['npz_sha256']==sha(ROOT/'m_gdsc_extra19.npz')
        result=dict(status='PASS',samples=19,m_shape=[19,225],auc_shape=[19,295],protected_candidate_overlap=0,old_train_parity_ids=r['sample_ids'].tolist(),m_parity_max_abs_difference=0.0,auc_parity_max_abs_difference=0.0,npz_sha256=qa['npz_sha256'],verifier_sha256=sha(Path(__file__)),training_performed=False)
    (ROOT/'verification_m_gdsc.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
