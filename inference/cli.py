"""JSON-lines agent adapter: python -m inference.cli --run-dir ..."""
import argparse
import json
import sys
from .agent_tool import DHCTool


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--history', choices=['none','train','train_val'], default='train')
    args = parser.parse_args()
    tool = DHCTool(args.run_dir, device=args.device, history=args.history)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = request.pop('action', 'rank_links')
            if action not in {'rank_links','observe'}: raise ValueError('Unsupported action')
            result = getattr(tool, action)(**request)
            print(json.dumps({'ok':True, 'result':result}, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({'ok':False,'error':str(exc)}), flush=True)

if __name__ == '__main__': main()
