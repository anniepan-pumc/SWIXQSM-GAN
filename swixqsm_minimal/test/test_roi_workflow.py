"""Frozen-prediction ROI CLI integration; synthetic volumes/checkpoint only."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from .. import evaluate
from ..util.quality import build_report
from ..util.nifti import sidecar_path
from ..util.data import read_manifest
from ..util.quality_gate import prepare_quality_rows
from ..util.reference import reference_identity


class ROIWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.csv'
        self.shape = (9, 10, 3)
        self.affine = np.diag([.8, .9, 3., 1.])
        self.rows = []
        self.rng = np.random.default_rng(420)
        labels = np.zeros(self.shape, dtype=np.int16)
        labels[:3] = 1
        labels[3:7] = 2
        self.checkpoint = self.root / 'frozen.pt'
        self.checkpoint.write_bytes(b'Synthetic frozen checkpoint; never a model to load/train')
        sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.selection = self.root / 'selection.json'
        self.selection.write_text(json.dumps({'schema': 'swixqsm.model_selection_record', 'version': 1,
            'checkpoint_path': str(self.checkpoint), 'checkpoint_sha256': sha,
            'selection_basis': 'fixed_before_test', 'test_feedback_used': False,
            'validation_used': False, 'source': 'Synthetic test-only authored record', 'evidence_files': []}))
        for i in range(7):
            person = i if i < 6 else 0
            target = np.full(self.shape, 10. * (person+1) + (4 if i == 6 else 0))
            target[labels == 2] += 25
            qsm_units = 'ppm' if i % 2 == 0 else 'ppb'
            row = {'subject_id': f'p{person}', 'global_subject_id': f'g{person}', 'case_id': f'case{i}',
                   'split': 'test', 'cohort': 'A' if person < 3 else 'B', 'qsm_units': qsm_units,
                   'qsm_reference': 'synthetic_reference', 'space_id': 'synthetic_space',
                   'roi_label_source': 'Synthetic integer labels; no real segmentation', 'roi_review': 'accepted',
                   'roi_reviewer': 'Synthetic software fixture', 'roi_review_date': '2026-10-17',
                   'prediction_units': 'ppb', 'prediction_reference': 'synthetic_reference'}
            arrays = {'swi_path': self.rng.uniform(1,100,self.shape),
                      'qsm_path': target / 1000 if qsm_units == 'ppm' else target,
                      'mask_path': np.ones(self.shape), 'swi_mask_path': np.ones(self.shape),
                      'roi_labels_path': labels, 'prediction_path': target+2}
            for role, array in arrays.items():
                row[role] = str(self.save(f'{i}_{role}.nii.gz', array))
            sidecar_path(row['prediction_path']).write_text(json.dumps({'Units':'ppb', 'ReferenceID':'synthetic_reference',
                'CheckpointSHA256':sha, 'DataQC':{'status':'unverified', 'reason':'Synthetic fixture'},
                'SpaceID':'synthetic_space'}))
            self.rows.append(row)
        # Only existence/identity metadata of train may be checked, never content.
        train = {key: '' for key in self.rows[0]}
        train.update(subject_id='train_only', global_subject_id='train_global', split='train', case_id='train_case',
                     qsm_units='ppm')
        for role in ('swi_path','qsm_path','mask_path','swi_mask_path','roi_labels_path'):
            path = self.root / f'unconsumed_train_{role}.nii.gz'
            path.write_bytes(b'Synthetic invalid image: must never be decoded in test evaluation')
            train[role] = str(path)
        self.all_rows = [train, *self.rows]
        self.write_manifest()
        self.spec = self.root / 'roi.json'
        self.spec.write_text(json.dumps({'schema':'swixqsm.roi_spec','version':1,'id':'known_synthetic_regions',
            'definition':'Software tests only, prespecified known labels', 'label_source':'Synthetic known integer arrays',
            'regions':[{'name':'left','labels':[1]},{'name':'bilateral','labels':[1,2]},
                       {'name':'absent','labels':[99]}], 'min_coverage':1.,'min_voxels':1}))

    def save(self, name, values):
        image = nib.Nifti1Image(np.asarray(values,dtype=np.float32),self.affine)
        image.header.set_xyzt_units('mm')
        path = self.root/name
        nib.save(image,path)
        return path

    def write_manifest(self):
        with self.manifest.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(self.rows[0]))
            writer.writeheader();writer.writerows(self.all_rows)

    def run_eval(self, output, *, roi=True, extra=()):
        args=['evaluate','--manifest',str(self.manifest),'--output',str(output),'--split','test',
              '--reference-id','synthetic_reference','--data-range','500','--selection-record',str(self.selection),
              '--allow-unverified-data','Synthetic fixture','--allow-unverified-independence','Synthetic identity declaration']
        if roi:
            args += ['--roi-spec',str(self.spec),'--roi-bootstrap-reps','40','--roi-bootstrap-seed','42']
        args += list(extra)
        with mock.patch.object(sys,'argv',args):
            evaluate.main()

    @staticmethod
    def csv_rows(path):
        with path.open() as f:
            return list(csv.DictReader(f))

    def test_roi_keeps_original_metrics_and_fixed_checkpoint_no_val_or_training(self):
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.glob('*.nii.gz')}
        checkpoint_bytes=self.checkpoint.read_bytes()
        plain=self.root/'plain';regional=self.root/'regional'
        self.run_eval(plain,roi=False)
        self.run_eval(regional,extra=('--distribution-diagnostics','--save-predictions'))
        self.assertEqual((plain/'per_case.csv').read_bytes(),(regional/'per_case.csv').read_bytes())
        self.assertEqual((plain/'per_subject.csv').read_bytes(),(regional/'per_subject.csv').read_bytes())
        self.assertEqual(json.loads((plain/'summary.json').read_text())['summary'],
                         json.loads((regional/'summary.json').read_text())['summary'])
        report=json.loads((regional/'roi_agreement.json').read_text())
        self.assertEqual(report['model_selection']['status'],'heldout_by_declaration')
        self.assertFalse(report['model_selection']['validation_used'])
        self.assertFalse(report['model_selection']['weight_updates_performed'])
        self.assertEqual(report['data_qc']['status'],'unverified')
        self.assertEqual(report['analysis_status'],'estimated')
        self.assertEqual(len(report['provenance']['cases']),7)
        self.assertEqual(self.checkpoint.read_bytes(),checkpoint_bytes)
        for path,sha in before.items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),sha)
        cases=self.csv_rows(regional/'roi_per_case.csv')
        subjects=self.csv_rows(regional/'roi_per_subject.csv')
        self.assertEqual(len(cases),7*3*2)
        self.assertEqual(len(subjects),6*3*2)
        p0=next(r for r in subjects if r['method']=='prediction' and r['region']=='left' and r['subject_id']=='p0')
        self.assertEqual(p0['n_cases_expected'],'2')
        self.assertAlmostEqual(float(p0['difference']),2,places=5)
        overall=next(r for r in report['agreement'] if r['method']=='prediction' and r['region']=='left' and r['group_type']=='overall')
        self.assertEqual(overall['n_participants_included'],6)
        self.assertAlmostEqual(overall['agreement']['estimates']['bias'],2,places=5)
        self.assertAlmostEqual(overall['agreement']['estimates']['pearson_r'],1,places=9)
        self.assertLess(overall['agreement']['estimates']['icc_a1'],1)
        by_cohort=[r for r in report['agreement'] if r['method']=='prediction' and r['region']=='left' and r['group_type']=='cohort']
        self.assertEqual({r['group']:r['n_participants_included'] for r in by_cohort},{'A':3,'B':3})
        absent=next(r for r in report['agreement'] if r['region']=='absent' and r['group_type']=='overall')
        self.assertEqual(absent['n_participants_included'],0)
        self.assertIsNone(absent['agreement']['estimates']['icc_a1'])
        self.assertTrue((regional/'roi_summary.csv').exists())
        self.assertTrue((regional/'distribution_diagnostics.json').exists())
        self.assertTrue((regional/'saved_predictions.csv').exists())
        # Same inputs/seed produce the same CI results; the output path is irrelevant.
        repeated=self.root/'repeat'
        self.run_eval(repeated)
        self.assertEqual(report['agreement'],json.loads((repeated/'roi_agreement.json').read_text())['agreement'])
        with self.assertRaises(SystemExit):self.run_eval(regional)

    def test_missing_maps_and_review_exclusions_deliver_explicit_counts(self):
        for row in self.rows:
            row['roi_labels_path']=''
        self.write_manifest()
        output=self.root/'missing'
        self.run_eval(output)
        report=json.loads((output/'roi_agreement.json').read_text())
        self.assertEqual(report['analysis_status'],'no_eligible_participants')
        self.assertTrue(all(r['status']=='missing_labelmap' for r in self.csv_rows(output/'roi_per_case.csv')))
        self.assertTrue(all(r['n_participants_included']==0 for r in report['agreement']))
        self.assertTrue(all(r['n_participants_expected'] in (3,6) for r in report['agreement']))

    def test_bad_labels_publish_no_partial_evaluation(self):
        values=nib.load(self.rows[-1]['roi_labels_path']).get_fdata()
        values[0,0,0]=1.5
        self.save(Path(self.rows[-1]['roi_labels_path']).name,values)
        output=self.root/'bad_labels'
        with self.assertRaisesRegex(ValueError,'integer'):
            self.run_eval(output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob('.swixqsm-evaluation-*')),[])

    def test_units_ppm_scaling_preserves_dimensionless_regional_agreement(self):
        ppb=self.root/'ppb';ppm=self.root/'ppm'
        self.run_eval(ppb)
        self.run_eval(ppm,extra=('--units','ppm'))
        def left(path):
            return next(r for r in json.loads((path/'roi_agreement.json').read_text())['agreement']
                        if r['method']=='prediction' and r['region']=='left' and r['group_type']=='overall')['agreement']['estimates']
        a,b=left(ppb),left(ppm)
        self.assertAlmostEqual(a['bias']/1000,b['bias'],places=12)
        self.assertAlmostEqual(a['loa_lower']/1000,b['loa_lower'],places=12)
        self.assertAlmostEqual(a['icc_a1'],b['icc_a1'],places=12)
        self.assertAlmostEqual(a['pearson_r'],b['pearson_r'],places=12)

    def test_bad_bootstrap_options_rejected_before_manifest_access(self):
        self.manifest.unlink()
        for extra in (('--roi-bootstrap-reps','-1'),('--roi-bootstrap-seed','-1')):
            with self.subTest(extra=extra),self.assertRaises(SystemExit):
                self.run_eval(self.root/'never',extra=extra)

    def test_verified_image_qc_does_not_upgrade_pending_roi_review(self):
        from .test_quality_workflow import QualityWorkflowTest
        fixture=QualityWorkflowTest('test_stale_report_and_selected_split_isolation')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        row=fixture.rows[-1]
        labels=np.ones(fixture.shape,dtype=np.int16)
        labels[:,20:]=2
        row.update(roi_labels_path=str(fixture.save('labels.nii.gz',labels)),
                   roi_label_source='Synthetic known anatomy',roi_review='pending',
                   roi_reviewer='',roi_review_date='')
        row['prediction_path']=str(fixture.save('fixed_prediction.nii.gz',nib.load(row['qsm_path']).get_fdata()))
        sidecar_path(row['prediction_path']).write_text(json.dumps({'Units':'ppm',
            'ReferenceID':'synthetic_reference','DataReference':reference_identity(fixture.contract['reference']),
            'DataQC':{'status':'verified'}}))
        fixture.write_manifest()
        def run(output):
            args=['evaluate','--manifest',str(fixture.manifest),'--output',str(output),
                  '--qc-report',str(fixture.report_path),'--reference-id','synthetic_reference',
                  '--roi-spec',str(self.spec),'--roi-bootstrap-reps','0']
            with mock.patch.object(sys,'argv',args):evaluate.main()
        stale=fixture.root/'stale_roi_qc'
        with self.assertRaisesRegex(ValueError,'QC changed'):
            run(stale)
        self.assertFalse(stale.exists())
        rows=prepare_quality_rows(read_manifest(fixture.manifest),fixture.manifest)
        report=build_report([r for r in rows if r['split']=='test'],fixture.contract)
        self.assertEqual(report['n_passed'],1)
        fixture.report_path.write_text(json.dumps(report,allow_nan=False))
        output=fixture.root/'verified_images_pending_roi'
        run(output)
        regional=json.loads((output/'roi_agreement.json').read_text())
        self.assertEqual(regional['data_qc']['status'],'verified')
        self.assertEqual(regional['analysis_status'],'no_eligible_participants')
        self.assertTrue(all(r['status']=='review_not_accepted' for r in self.csv_rows(output/'roi_per_case.csv')))


if __name__=='__main__':
    unittest.main()
