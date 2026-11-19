import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,SpreadsheetFile,Workbook} from '@oai/artifact-tool';
const C=path.dirname(fileURLToPath(import.meta.url)),B=process.env.TEMPLATE_RUN_DIR||path.join(path.dirname(C),'generated');
const out=process.env.RESULT_OUTPUT_DIR||path.join(B,'xlsx');
const names=process.argv.slice(2);if(!names.length)names.push('result1.xlsx','result2.xlsx','result3.xlsx','result4-2.xlsx','result4-3.xlsx','第三问_全年365天_八种情况.xlsx');
const date=x=>x===null?null:new Date(x+'T00:00:00.000Z');
function extendTemplateStyle(sheet,style,groups,lastRow,lastCol){
 const body=sheet.getRange(`A2:${lastCol}${lastRow}`);body.clear({applyTo:'formats'});
 for(let j=0;j<style.rows[0].length;j++){
  const c=style.rows[0][j],col=String.fromCharCode(65+j),r=sheet.getRange(`${col}2:${col}${lastRow}`);
  r.format.font=c.font;r.format.horizontalAlignment=c.horizontal;r.format.verticalAlignment=c.vertical;
  const edges={};for(const edge of ['left','right'])if(c.borders[edge])edges[edge]={style:c.borders[edge],color:'#000000'};
  r.format.borders=edges;
 }
 body.format.rowHeight=style.heights[0];
 for(const g of groups){
  const end=g.start_row+g.rows-1;
  sheet.getRange(`A${g.start_row}:${lastCol}${g.start_row}`).format.borders={top:{style:'thin',color:'#000000'}};
  sheet.getRange(`A${end}:${lastCol}${end}`).format.borders={bottom:{style:'thin',color:'#000000'}};
  if(g.rows===style.heights.length)for(let j=0;j<g.rows;j++)if(style.heights[j]!==style.heights[0])sheet.getRange(`A${g.start_row+j}:${lastCol}${g.start_row+j}`).format.rowHeight=style.heights[j];
  for(let i=1;i<g.rows;i++)for(let j=0;j<style.rows[0].length;j++){
   const index=style.rows.length===3?(i===g.rows-1?2:1):i;
   const c=style.rows[index][j],base=style.rows[0][j];if(c.horizontal===base.horizontal&&c.vertical===base.vertical)continue;
   const cell=sheet.getRange(`${String.fromCharCode(65+j)}${g.start_row+i}`);cell.format.horizontalAlignment=c.horizontal;cell.format.verticalAlignment=c.vertical;
  }
 }
}
async function verifyAndSave(w,name,ranges,presentation){
 if(presentation)for(const [name,p] of Object.entries(presentation)){
  const s=w.worksheets.getItem(name);s.getRangeByIndexes(0,0,p.rows,p.cols).format.numberFormat=p.number_formats;
  for(const c of p.widths)s.getRangeByIndexes(0,c.first_column-1,1,c.last_column-c.first_column+1).format.columnWidth=c.width;
  for(const r of p.heights)s.getRangeByIndexes(r.first_row-1,0,r.last_row-r.first_row+1,p.cols).format.rowHeight=r.height;
 }
 w.recalculate();const inspections=[];
 for(const [sheet,range] of ranges){
  inspections.push({sheet,range,inspection:(await w.inspect({kind:'region',sheetId:sheet,range,maxChars:1000,tableMaxRows:3,tableMaxCols:4})).ndjson});
  const blob=await w.render({sheetName:sheet,range,scale:1.4,format:'png'});
  await fs.writeFile(path.join(B,'预览',`${name}_${sheet}_${range.replace(':','-')}.png`),new Uint8Array(await blob.arrayBuffer()));
 }
 await fs.writeFile(path.join(B,'核验',`${name}.artifact检查.json`),JSON.stringify(inspections,null,2));
 await fs.mkdir(out,{recursive:true});const result=await SpreadsheetFile.exportXlsx(w);await result.save(path.join(out,name));console.log('EXPORTED',name);
}
for(const name of names){
 const p=JSON.parse(await fs.readFile(path.join(B,'核验',`${name}.payload.json`),'utf8'));
 if(name.startsWith('第三问')){
  const w=Workbook.create();const summary=w.worksheets.add('全年汇总');const ranges=[['全年汇总','A1:G10']];
  const caseOrder=['000','100','010','110','001','101','011','111'];
  for(const key of caseOrder)w.worksheets.add('方案'+key);const cal=w.worksheets.add('参数校准');
  summary.getRange('A1:G1').values=[['方案','全年费用/元','正式期费用/元','正式期紧急费/元','年初储电量/kWh','年末储电量/kWh','正式余量参数']];
  let row=2;
  for(const key of caseOrder){const rows=p.cases[key];
   const s=w.worksheets.getItem('方案'+key);s.getRange('A1:O1').values=[p.headers];
   s.getRange('A2:O366').values=rows.map(r=>[date(r[0]),...r.slice(1)]);
   s.getRange('M2:M366').formulas=rows.map((_,i)=>[`=SUM(I${i+2}:L${i+2})`]);
   s.getRange('A2:A366').setNumberFormat('yyyy-mm-dd');s.getRange('B2:H366').setNumberFormat('#,##0.0000');s.getRange('I2:M366').setNumberFormat('#,##0.00');s.getRange('O2:O366').setNumberFormat('0.00');
   summary.getRange(`A${row}`).values=[[`(${key.split('').join(',')})`]];summary.getRange(`B${row}:G${row}`).formulas=[[`=SUM('方案${key}'!M2:M366)`,`=SUM('方案${key}'!M33:M366)`,`=SUM('方案${key}'!L33:L366)`,`='方案${key}'!B2`,`='方案${key}'!C366`,`='方案${key}'!O33`]];row++;
   ranges.push(['方案'+key,'A1:G9'],['方案'+key,'H1:O9']);
  }
  cal.getRange('A1:G1').values=[['方案','候选参数','初始储电量/kWh','末尾储电量/kWh','回放费用/元','库存校正评分/元','选中']];
  cal.getRange('J1:K1').values=[['库存估值系数',p.inventory_value]];
  cal.getRange('A2:G97').values=p.calibration.map(r=>[...r.slice(0,6),r[6]?'是':'否']);
  cal.getRange('F2:F97').formulas=p.calibration.map((_,i)=>[`=E${i+2}-$K$1*(D${i+2}-C${i+2})`]);
  cal.getRange('C2:F97').setNumberFormat('#,##0.00');cal.getRange('B2:B97').setNumberFormat('0.00');
  summary.getRange('B2:D9').setNumberFormat('#,##0.00');summary.getRange('E2:F9').setNumberFormat('#,##0.0000');summary.getRange('G2:G9').setNumberFormat('0.00');
  for(let i=0;i<10;i++){
   const s=w.worksheets.getItemAt(i);const used=s.getUsedRange();used.format.font={name:'Microsoft YaHei',size:10};used.format.rowHeight=21;used.format.columnWidth=19;used.format.verticalAlignment='center';
   const last=i===0?'G':i===9?'K':'O';s.getRange(`A1:${last}1`).format.font={bold:true};s.getRange(`A1:${last}1`).format.wrapText=true;s.getRange(`A1:${last}1`).format.rowHeight=36;
   s.freezePanes.freezeRows(1);s.showGridLines=false;
  }
  ranges.push(['参数校准','A1:G9']);await verifyAndSave(w,name,ranges);continue;
 }
 // The supplied blank template avoids stale shared-formula metadata on a changed year-end total.
 const source=!p.alignment&&process.env.TEMPLATE_SOURCE_DIR?path.join(process.env.TEMPLATE_SOURCE_DIR,name):path.join(C,'reproduction/C题/附件/附件5',name);
 const w=await SpreadsheetFile.importXlsx(await FileBlob.load(source));
 const plan=w.worksheets.getItem('计划购电量');const storage=w.worksheets.getItem('充放电量');
 if(name==='result1.xlsx'){
  plan.getRange('B2:B145').values=p.plan;storage.getRange('B2:C7').values=p.storage;storage.getRange('E2:E3').values=p.endpoints;
  await verifyAndSave(w,name,[['计划购电量','A1:B10'],['计划购电量','A140:B145'],['充放电量','A1:E7']],p.presentation);continue;
 }
 // Original headers and date column remain exactly as provided in Attachment 5.
 const totalFormula=i=>p.alignment&&i===333?'=IF(COUNT(B335:EO335)=144,SUM(B335:EO335),"")':`=SUM(B${i+2}:EO${i+2})`;
 plan.getRange('B2:EO335').values=p.plan;plan.getRange('EP2:EP335').formulas=p.dates.map((_,i)=>[totalFormula(i)]);plan.getRange('EQ2:EQ335').values=p.plan_fees.map(x=>[x]);
 const ranges=[['计划购电量','A1:F7'],['计划购电量','EL1:EQ7']];
 if(name==='result3.xlsx'||name==='result4-3.xlsx'){
  const s=w.worksheets.getItem('调整购电量');s.getRange('B2:EO335').values=p.normal;s.getRange('EP2:EP335').formulas=p.dates.map((_,i)=>[totalFormula(i)]);s.getRange('EQ2:EQ335').values=p.adjustment_fees.map(x=>[x]);ranges.push(['调整购电量','A1:F7'],['调整购电量','EL1:EQ7']);
 }
 extendTemplateStyle(storage,p.storage_style,p.dates.map((_,i)=>({start_row:2+6*i,rows:6})),2005,'F');
 storage.getRange('A2:F2005').values=p.storage.map(r=>[date(r[0]),...r.slice(1)]);
 storage.getRange('A2:F2005').format.numberFormat=p.storage_formats;
 const emergency=w.worksheets.getItem('紧急购电量');
 extendTemplateStyle(emergency,p.event_style,p.event_groups,p.events.length+1,'C');
 emergency.getRange(`A2:C${p.events.length+1}`).values=p.events.map(r=>[date(r[0]),r[1],r[2]]);
 emergency.getRange(`A2:C${p.events.length+1}`).format.numberFormat=p.event_formats;
 ranges.push(['充放电量','A1:F13'],['充放电量','A1994:F2005'],['紧急购电量','A1:C13']);
 if(p.alignment){ranges.push(['计划购电量','EM332:EQ335']);if(name==='result3.xlsx'||name==='result4-3.xlsx')ranges.push(['调整购电量','EM332:EQ335']);}
 await verifyAndSave(w,name,ranges,p.presentation);
}
