"""读取问题一附件和结果模板，整理单日 144 时段输入数据。"""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
from openpyxl import load_workbook

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
for name in ('data', 'results', 'figures', 'latex'):
    (BASE / name).mkdir(parents=True, exist_ok=True)
source = BASE / 'data/source/附件1.xlsx'
template = BASE / 'data/source/result1_template.xlsx'
if not source.exists():
    source = ROOT / '附件/附件1.xlsx'
if not template.exists():
    template = ROOT / '附件/附件5/result1.xlsx'
wb = load_workbook(source, read_only=True, data_only=True)
rows = list(wb.active.values)
wb.close()
def minutes(value):
    if isinstance(value, (dt.time, dt.datetime)):
        return value.hour * 60 + value.minute
    s = str(value).strip()
    hm = s.replace('+1', '').split(':')
    return int(hm[0]) * 60 + int(hm[1]) + (1440 if '+1' in s else 0)
def label(minute):
    return f'{minute // 60:02d}:{minute % 60:02d}'
assert len(rows) == 145
records = []
for i, row in enumerate(rows[1:], 1):
    end = minutes(row[0])
    assert end == i * 10
    p, load, pv = map(float, row[1:4])
    assert p > 0 and load >= 0 and pv >= 0
    records.append(dict(slot=i, source_time=str(row[0]), start_minute=end-10,
                        end_minute=end, interval=f'{label(end-10)}-{label(end)}',
                        price_yuan_kwh=p, load_kw=load, pv_forecast_kw=pv))
with (BASE / 'data/input_day.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
wb = load_workbook(template, read_only=True, data_only=True)
labels = [r[0] for r in list(wb['计划购电量'].values)[1:]]
wb.close()
assert len(labels) == 144
mapping = [dict(excel_row=i+2, original_template_interval=old,
                corrected_interval=r['interval'], source_sample=r['source_time'])
           for i, (old, r) in enumerate(zip(labels, records))]
with (BASE / 'data/time_label_mapping.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(mapping[0]))
    writer.writeheader()
    writer.writerows(mapping)
metadata = dict(source='附件/附件1.xlsx', template='附件/附件5/result1.xlsx',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
                n=144, interval_hours=1/6,
                convention='Right-endpoint rectangular approximation: 00:10 sample represents 00:00-00:10.',
                original_template_first=labels[0], original_template_last=labels[-1],
                missing_numeric_values=0, negative_values=0)
(BASE / 'data/input_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(metadata, ensure_ascii=False, indent=2))
