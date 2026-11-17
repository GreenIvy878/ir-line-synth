/* Records without real data: the parametric section of a recipe -> preview of the first record -> saved records. */
'use strict';
const recordsUI={text:null,profiles:[],seed:17,split:'train',directory:'synthetic_parametric',busy:false,
  preview:null,message:''};
const recordsZoom=6;

function recordsMessage(text,error=false) {
  recordsUI.message=text;
  if($('records-message')) {$('records-message').textContent=text;$('records-message').classList.toggle('error',error);}
}
async function openRecords() {
  if(recordsUI.text===null) {
    const defaults=await request('/api/synthetic/parametric-default');
    recordsUI.text=defaults.text;recordsUI.profiles=defaults.profiles;
  }
  renderRecords();
  if(!$('records-dialog').open)$('records-dialog').showModal();
}
function renderRecords() {
  const ui=recordsUI;
  $('records-dialog').innerHTML=`<div class="autogen-heading"><h2 id="records-title">Записи без реальных данных</h2><button id="records-close" aria-label="Закрыть записи без реальных данных">Закрыть</button></div>
    <p>Запись линейки строится целиком из модели датчика: объекты по описанию (форма, поверхность, движение) проходят над пустой лентой профиля, сигнал проходит шум и АЦП, кадры нарезаются как в наборе, рамки — по геометрии объектов. Описание ниже — раздел <code>parametric</code> рецепта (docs/recipes.md); тот же запуск можно выполнить командой <code>ir-line-synth generate</code>.</p>
    <div id="records-message" class="synthetic-message" role="status" aria-live="polite"></div>
    <div class="autogen-layout records-layout"><section class="card"><h3>Описание записей</h3><div class="autogen-rules">
      <label class="records-field">Раздел parametric (YAML)<textarea id="records-text" spellcheck="false" wrap="off" rows="28">${esc(ui.text)}</textarea></label>
      <p class="muted">Профили из поставки: ${esc(ui.profiles.join(', '))}. Путь к своему профилю — от папки, где запущен редактор.</p>
      ${synNumber('records-seed','Seed',ui.seed,0,4294967295,1)}
      <div class="controls"><button id="records-preview" class="primary">Предпросмотр первой записи</button><button id="records-default">Вернуть пример</button></div>
      <h4>Сохранение</h4>
      <label>Часть набора <select id="records-split"><option value="train">Train</option><option value="val">Validation</option><option value="test">Test</option></select></label>
      <label class="records-field">Папка вывода <input id="records-directory" type="text" value="${esc(ui.directory)}"></label>
      <div class="controls"><button id="records-save">Построить и сохранить все записи</button></div>
      <p id="records-save-status" class="records-status" role="status" aria-live="polite"></p>
      <p class="muted">Сохраняются коды каждой записи (records/&lt;часть&gt;/), её рамки и описание, а кадры — как примеры части (PNG + YOLO). Команда <code>ir-line-synth verify</code> строит их заново без исходного набора.</p>
    </div></section>
    <section><h3>Кадры первой записи</h3><p id="records-info" class="muted"></p>
      <div id="records-frames" class="autogen-gallery records-gallery"></div>
      <details class="records-recipe"><summary>Этот запуск как рецепт</summary><textarea id="records-recipe" readonly spellcheck="false" rows="16"></textarea></details>
    </section></div>`;
  $('records-split').value=ui.split;
  $('records-close').onclick=()=>$('records-dialog').close();
  $('records-text').oninput=()=>ui.text=$('records-text').value;
  $('records-seed').onchange=()=>ui.seed=Number($('records-seed').value);
  $('records-split').onchange=()=>ui.split=$('records-split').value;
  $('records-directory').onchange=()=>ui.directory=$('records-directory').value;
  $('records-default').onclick=()=>recordsAction(async()=>{
    const defaults=await request('/api/synthetic/parametric-default');
    ui.text=defaults.text;$('records-text').value=ui.text;recordsMessage('Описание заменено примером.');
  });
  $('records-preview').onclick=()=>recordsAction(recordsPreview);
  $('records-save').onclick=()=>recordsAction(recordsSave);
  recordsMessage(ui.message||'Измените описание при необходимости и постройте предпросмотр.');
  recordsFrames();recordsControls();
}
async function recordsAction(action) {
  if(recordsUI.busy)return;
  recordsUI.busy=true;recordsControls();
  try {await action();} catch(error) {recordsMessage(error.message,true);}
  finally {recordsUI.busy=false;recordsControls();}
}
function recordsControls() {
  if(!$('records-preview'))return;
  document.querySelectorAll('#records-dialog input, #records-dialog select, #records-dialog button, #records-text')
    .forEach(node=>node.disabled=recordsUI.busy);
  $('records-close').disabled=false;
}
function recordsDraw(canvas,frame,boxes) {
  const ctx=canvas.getContext('2d'),rows=frame.length,columns=frame[0].length;
  for(let y=0;y<rows;y++)for(let x=0;x<columns;x++) {
    const gray=frame[y][x];
    ctx.fillStyle=`rgb(${gray},${gray},${gray})`;ctx.fillRect(x*recordsZoom,y*recordsZoom,recordsZoom,recordsZoom);
  }
  ctx.lineWidth=1;
  for(const item of boxes) {
    const [x1,y1,x2,y2]=item.box;
    ctx.strokeStyle=item.cut?'#e5484d':'#f5a524';
    ctx.setLineDash(item.cut?[4,3]:[]);
    ctx.strokeRect(x1*recordsZoom+0.5,y1*recordsZoom+0.5,(x2-x1)*recordsZoom-1,(y2-y1)*recordsZoom-1);
  }
}
function recordsFrames() {
  const preview=recordsUI.preview;
  if(!preview) {
    $('records-frames').innerHTML='<p class="empty">После предпросмотра здесь появятся кадры первой записи с рамками (1 пиксель кадра = 6 точек).</p>';
    $('records-info').textContent='';$('records-recipe').value='';
    return;
  }
  const rows=preview.frames[0]?.length||0,columns=preview.frames[0]?.[0]?.length||0;
  $('records-frames').innerHTML=preview.frames.map((_,index)=>`<article class="card autogen-candidate">
    <p>Кадр ${index+1} · объектов ${preview.boxes[index].length}</p>
    <canvas id="records-frame-${index}" width="${columns*recordsZoom}" height="${rows*recordsZoom}" aria-label="Кадр ${index+1} первой записи"></canvas></article>`).join('');
  preview.frames.forEach((frame,index)=>recordsDraw($(`records-frame-${index}`),frame,preview.boxes[index]));
  const names={};preview.names.forEach(name=>names[name]=(names[name]||0)+1);
  $('records-info').textContent=`Запись 1 из ${preview.records}: ${preview.objects} объектов (${Object.entries(names).map(([name,count])=>`${name} ${count}`).join(', ')}), скорость ленты ${preview.speed_mm_s.toFixed(1)} мм/с, ${preview.frames.length} кадров. Оранжевые рамки — объекты целиком в кадре, красные пунктирные — обрезанные краем кадра.`+
    (preview.normalized?'':' Кадры этого профиля не нормированы (коды): показаны растянутыми, сохранить их как PNG нельзя.');
  $('records-recipe').value=preview.recipe;
}
async function recordsPreview() {
  const ui=recordsUI;
  recordsMessage('Строится первая запись…');
  ui.preview=await synPost('parametric-preview',{text:ui.text,seed:ui.seed,split:ui.split});
  recordsFrames();
  const notes=ui.preview.notes.length?' '+ui.preview.notes.join(' '):'';
  recordsMessage(`Готово: первая запись из ${ui.preview.records}.${notes} Сохранение построит все записи с этим seed.`);
}
// The result of saving is shown next to the save button as well: the message at the top is out of sight there.
function recordsSaveStatus(text,error=false) {
  recordsMessage(text,error);
  $('records-save-status').textContent=text;$('records-save-status').classList.toggle('error',error);
}
async function recordsSave() {
  const ui=recordsUI;
  recordsSaveStatus('Строятся и сохраняются записи… Это может занять время.');
  try {
    const result=await synPost('parametric-save',{text:ui.text,seed:ui.seed,split:ui.split,directory:ui.directory});
    const frames=result.saved.reduce((sum,row)=>sum+row.frames,0);
    const skipped=result.skipped.length?` Не сохранено ${result.skipped.length}: ${result.skipped.map(row=>`запись ${row.index+1} — ${row.reason}`).join('; ')}.`:'';
    recordsSaveStatus(`Сохранено записей ${result.saved.length} из ${result.requested}, кадров ${frames} в ${ui.directory} (${ui.split}).${skipped}`,!!result.skipped.length);
  } catch(error) {recordsSaveStatus(error.message,true);}
}
$('records-open').onclick=()=>openRecords().catch(error=>alert(error.message));
