#!/usr/bin/env python3
import os, sys, time, glob, subprocess, json

def get_tb_metrics(tag):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        fs = sorted(glob.glob(f"models/{tag}/tb/*/events*"))
        if not fs:
            return None
        ea = EventAccumulator(fs[0])
        ea.Reload()
        res = {}
        for k in ['train/std', 'train/value_loss', 'rollout/ep_rew_mean', 'rollout/ep_len_mean', 'time/fps']:
            if k in ea.Tags().get('scalars', []):
                ev = ea.Scalars(k)
                if ev: res[k] = (ev[-1].step, ev[-1].value)
        if len(fs) > 1:
            ea2 = EventAccumulator(fs[1])
            ea2.Reload()
            if 'eval/difficulty' in ea2.Tags().get('scalars', []):
                diffs = ea2.Scalars('eval/difficulty')
                succs = ea2.Scalars('eval/success_rate')
                if diffs:
                    res['eval/difficulty'] = diffs[-1].value
                    res['eval/success_rate'] = succs[-1].value
        return res
    except Exception as e:
        return None

def is_running(tag):
    cmd = f"pgrep -f 'train_ppo.py.*{tag}'"
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    return res.returncode == 0, res.stdout.strip()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="p2_v24")
    parser.add_argument("--max-seconds", type=int, default=480)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== Monitor started for {args.tag} (max wait {args.max_seconds}s) ===")
    
    last_print = 0
    while True:
        elapsed = time.time() - t_start
        running, pids = is_running(args.tag)
        
        metrics = get_tb_metrics(args.tag)
        step = metrics.get('train/std', (0, 0))[0] if metrics else 0
        std = metrics.get('train/std', (0, 0))[1] if metrics else 0.0
        v_loss = metrics.get('train/value_loss', (0, 0))[1] if metrics else 0.0
        fps = metrics.get('time/fps', (0, 0))[1] if metrics else 0.0
        diff = metrics.get('eval/difficulty', 0.20) if metrics else 0.20
        succ = metrics.get('eval/success_rate', 0.0) if metrics else 0.0

        if time.time() - last_print >= 20 or not running:
            last_print = time.time()
            m = int(elapsed // 60)
            s = int(elapsed % 60)
            print(f"[{m:02d}:{s:02d}] Step: {step:>7} / 2,000,000 ({step/2000000*100:5.1f}%) | "
                  f"diff: {diff:.2f} | eval_succ: {succ*100:5.1f}% | "
                  f"v_loss: {v_loss:6.2f} | std: {std:.3f} | FPS: {fps:.0f}")

        # Check if training finished
        results_file = f"models/{args.tag}/results.json"
        if not running:
            time.sleep(2) # grace period for files to flush
            if os.path.exists(results_file):
                print(f"\n🎉 Training {args.tag} completed successfully!")
                with open(results_file) as f:
                    print("Results summary:\n", f.read())
                
                print("\n=== Running eval_phase.py on completed model ===")
                eval_cmd = f"python3 eval_phase.py --model models/{args.tag}/model.zip --phase 2 --episodes 50 --omega-th 1.2"
                eval_proc = subprocess.run(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                print(eval_proc.stdout)
                
                print("=== TRAINING_AND_EVALUATION_COMPLETED ===")
                sys.exit(0)
            else:
                print(f"⚠️ Process {args.tag} is not running, but {results_file} not found. Check train_{args.tag}.log:")
                log_file = f"results/train_{args.tag}.log"
                if os.path.exists(log_file):
                    subprocess.run(f"tail -n 25 {log_file}", shell=True)
                sys.exit(1)

        if elapsed >= args.max_seconds:
            print(f"\n=== CHUNK_COMPLETED_STILL_TRAINING: Step {step} / 2,000,000 ({step/2000000*100:.1f}%) ===")
            sys.exit(0)

        time.sleep(5)

if __name__ == "__main__":
    main()
