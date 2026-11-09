#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dhc-csv', required=True)
    p.add_argument('--tgn-external-csv', default='')
    p.add_argument('--tgb2-custom-csv', default='')
    p.add_argument('--tgb2-official-csv', default='')
    p.add_argument('--dyglib-custom-csv', default='')
    p.add_argument('--out-csv', required=True)
    args = p.parse_args()

    dhc = pd.read_csv(args.dhc_csv)
    extra_rows = []

    if args.tgn_external_csv:
        ext = pd.read_csv(args.tgn_external_csv)
        ext_old = ext[ext['split'] == 'old_nodes']
        if len(ext_old) > 0:
            ext_old = ext_old.iloc[0]
            extra_rows.append(
                {
                    'model': 'tgn_external_old_nodes',
                    'test_mrr': None,
                    'test_auc': float(ext_old['auc']),
                    'test_ap': float(ext_old['ap']),
                }
            )

    if args.tgb2_custom_csv:
        tgb2 = pd.read_csv(args.tgb2_custom_csv)
        tgb2_test = tgb2[tgb2['split'] == 'test']
        if len(tgb2_test) > 0:
            tgb2_test = tgb2_test.iloc[0]
            extra_rows.append(
                {
                    'model': 'tgb2_custom_tgn',
                    'test_mrr': None,
                    'test_auc': float(tgb2_test['auc']),
                    'test_ap': float(tgb2_test['ap']),
                }
            )

    if args.tgb2_official_csv:
        tgb2_official = pd.read_csv(args.tgb2_official_csv)
        tgb2_official_test = tgb2_official[tgb2_official['split'] == 'test']
        if len(tgb2_official_test) > 0:
            tgb2_official_test = tgb2_official_test.iloc[0]
            extra_rows.append(
                {
                    'model': 'tgb2_official_tgn',
                    'test_mrr': float(tgb2_official_test['mrr']),
                    'test_auc': None,
                    'test_ap': None,
                }
            )

    if args.dyglib_custom_csv:
        dyglib = pd.read_csv(args.dyglib_custom_csv)
        dyglib_test = dyglib[dyglib['split'] == 'test']
        if len(dyglib_test) > 0:
            dyglib_test = dyglib_test.iloc[0]
            extra_rows.append(
                {
                    'model': 'dyglib_custom_tgn',
                    'test_mrr': None,
                    'test_auc': float(dyglib_test['auc']),
                    'test_ap': float(dyglib_test['ap']),
                }
            )

    out = pd.concat([dhc, pd.DataFrame(extra_rows)], ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(out)


if __name__ == '__main__':
    main()
