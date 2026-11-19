"""按依赖顺序调用当前模型；不运行历史网格对比、论文分析或旧导出程序。"""
from pathlib import Path
import argparse
import os
import subprocess
import sys

C = Path(__file__).resolve().parent
P = C / 'reproduction'
STAGES = ('inputs', 'forecast', 'q1', 'q2', 'q3', 'price', 'q4', 'prepare')


def run(arguments, cwd=C, code=None):
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    if code is not None:
        env['PYTHONPATH'] = str(code) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, '-u', '-B', *map(str, arguments)],
                   cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description='从原附件重建并运行最新四问模型。')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES),
                        help='依次运行指定阶段；后续阶段需要前置阶段的数据。')
    parser.add_argument('--list', action='store_true', help='只列出运行顺序。')
    args = parser.parse_args()
    if args.list:
        print(' -> '.join(args.stages))
        return
    for stage in args.stages:
        print(f'开始：{stage}', flush=True)
        if stage == 'inputs':
            run([C / 'rebuild_inputs.py'])
        elif stage == 'forecast':
            run([C / 'retrain_components.py'])
        elif stage == 'q1':
            run([P / 'q1/scripts/prepare_input.py'])
            run([P / 'q1/scripts/solve_q1.py'])
        elif stage == 'q2':
            run(['-m', 'optimization.run_planning'], P / 'q2', P / 'q2/code')
        elif stage == 'q3':
            run(['-m', 'optimization.run_cases'], P / 'q3', P / 'q3/code')
        elif stage == 'price':
            run([P / 'q4/code/price_forecast.py'], P / 'q4', P / 'q4/code')
        elif stage == 'q4':
            for method in ('weekly_baseline', 'xgb_frozen', 'xgb_adaptive'):
                run([C / 'run_q4_worker.py', method])
            run([C / 'merge_q4_workers.py'])
        else:
            env = os.environ.copy()
            env['ALIGN_TEMPLATE_TIMES'] = '1'
            env['TEMPLATE_RUN_DIR'] = str(C.parent / 'generated')
            env['TEMPLATE_SOURCE_DIR'] = str(C.parent / 'results')
            subprocess.run([sys.executable, '-u', '-B', str(C / 'prepare_template_payload.py')],
                           cwd=C, env=env, check=True)
        print(f'完成：{stage}', flush=True)
    print('所选阶段完成。Excel 导出命令见 README。')


if __name__ == '__main__':
    main()
