/* Object generator: new objects from real ones -> hand-picked -> editor object bank. */
'use strict';
const morphKinds={blob:'Обычный блоб',rolling:'Катающийся',swinging:'Качающийся (рваный)',long:'Длинный',short:'Короткий'};
const morphUI={kind:'blob',split:'train',sources:'real',count:10,seed:1,ratioMin:1.25,ratioMax:5,hideRejected:false,busy:false,
  limits:{noise_max:0.3,hotspot_max:0.5,stripes_max:0.2,dropout_max:0.15},
  candidates:[],selected:new Set(),added:new Set(),library:null,librarySplit:null,message:''};
const morphLimitFields=[['morph-noise','noise_max','Шум пикселей'],['morph-hotspot','hotspot_max','Смещение яркого центра к краю'],
  ['morph-stripes','stripes_max','Полосы каналов'],['morph-dropout','dropout_max','Выбитые пиксели']];
const morphZoom=8,morphCell=26;

function morphReason(filter) {
  if(filter.accepted)return 'фильтр: похож на реальные';
  const [code,detail]=filter.reason.split(':');
  return {kind:`фильтр: читается как «${morphKinds[detail]||detail}»`,range:`фильтр: вне диапазона (${detail})`,
    unlike_real:'фильтр: не похож на реальные',copy:'фильтр: почти копия исходника',empty:'фильтр: пусто'}[code]||filter.reason;
}
// Real objects of the bank that can be sources: of the chosen split and not stubs cut by the frame edge.
function morphBankSources() {
  return syntheticUI.objectBank.filter(object=>!object.morph&&!object.clipped&&object.source.split===morphUI.split)
    .map(object=>({source:synCopy(object.source),box_index:object.box_index}));
}
// What the server did to the smallest objects of the batch (morph.sift_small).
function morphSifting(report) {
  if(!report)return '';
  if(!report.applied)return ' Отсев мелких не делался: в партии меньше 10 объектов.';
  const faint=report.removed_dim!==undefined;
  const lists=`самые короткие и самые узкие, по ${report.removed_short}`+(faint?`; самые мелкие по площади — ${report.removed_small}, самые тусклые — ${report.removed_dim}`:'');
  const floor=`не короче ${report.min_length} строк и не уже ${report.min_width} датчиков`+
    (faint?`, площадь от ${report.min_area} пикселей, средняя яркость от ${report.min_brightness.toFixed(3)}`:'');
  return ` Отсев мелких: убрано ${report.removed} (${lists}), заменено ${report.replaced}; теперь все ${floor}.`;
}
function morphMessage(text,error=false) {
  morphUI.message=text;
  if($('morph-message')) {$('morph-message').textContent=text;$('morph-message').classList.toggle('error',error);}
}
function openMorphGenerator() {
  renderMorphGenerator();
  if(!$('morph-dialog').open)$('morph-dialog').showModal();
}
function renderMorphGenerator() {
  const ui=morphUI;
  $('morph-dialog').innerHTML=`<div class="autogen-heading"><h2 id="morph-title">Генератор объектов</h2><button id="morph-close" aria-label="Закрыть генератор объектов">Закрыть</button></div>
    <p>Новые объекты строятся из реальных: меняется то, как яйцо прошло под датчиками (скорость, снос поперёк ленты, качание), затем накладываются несовершенства. Отметьте понравившиеся и добавьте их в набор объектов. Дальше они работают как обычные объекты: ручная сцена, вариации, автогенератор.</p>
    <div id="morph-message" class="synthetic-message" role="status" aria-live="polite"></div>
    <div class="autogen-layout"><section class="card"><h3>Что генерировать</h3><div class="autogen-rules">
      <label>Тип объекта <select id="morph-kind">${Object.entries(morphKinds).map(([key,name])=>`<option value="${key}">${name}</option>`).join('')}</select></label>
      <label>Часть набора <select id="morph-split">${synSplitOptions()}</select></label>
      <label>Исходники <select id="morph-sources"><option value="real">Подходящие целые реальные объекты части</option><option value="bank">Реальные объекты набора редактора из этой части</option></select></label>
      <p class="muted">Источники, статистика фильтра и новые объекты — только этой части. Обрубки (объекты, срезанные краем кадра) источниками не бывают. Из партии от 10 объектов 10 % самых коротких и 10 % самых узких заменяются новыми; у катающихся и качающихся ещё 10 % самых мелких по площади и 20 % самых тусклых (средняя яркость пикселей объекта).</p>
      ${synNumber('morph-count','Количество',ui.count,1,1000,1)}
      ${synNumber('morph-seed','Seed',ui.seed,0,4294967295,1)}
      <h4>Блоб: отношение длины (по времени) к ширине</h4>
      ${synNumber('morph-ratio-min','От',ui.ratioMin,0.1,20,0.05)}
      ${synNumber('morph-ratio-max','До',ui.ratioMax,0.1,20,0.05)}
      <p class="muted">Только для типа «Обычный блоб». Объекты вне диапазона не показываются, их место занимают новые. У реальных блобов медиана 2,0, а 5–95 % между 1,5 и 3,0.</p>
      <h4>Несовершенства: сила до (0–1)</h4>
      ${morphLimitFields.map(([id,key,label])=>synNumber(id,label,ui.limits[key],0,1,0.05)).join('')}
      <p class="muted">Сила каждого эффекта выбирается случайно от 0 до заданной, поэтому в партии есть и мягкие, и сильные варианты.</p>
      <label>Скрыть отбракованные фильтром <input id="morph-hide" type="checkbox" ${ui.hideRejected?'checked':''}></label>
      <div class="controls"><button id="morph-generate" class="primary">Сгенерировать</button><button id="morph-more">Ещё (seed + 1)</button></div>
      <p id="morph-library" class="muted"></p>
    </div></section>
    <section><h3>Кандидаты</h3><p class="muted">Фильтр только подсказывает: он сравнивает объект с реальными того же типа. Решаете вы.</p>
      <div class="controls"><button id="morph-select-all">Выбрать все</button><button id="morph-select-none">Снять выбор</button><button id="morph-deselect-rejected">Снять выбор с отбракованных фильтром</button><span id="morph-count-info"></span></div>
      <div id="morph-results" class="autogen-gallery morph-gallery"></div>
      <div class="controls morph-add"><button id="morph-add" class="primary">Добавить выбранные в набор</button><span id="morph-added-info" class="muted"></span></div>
    </section></div>`;
  $('morph-kind').value=ui.kind;$('morph-sources').value=ui.sources;$('morph-split').value=ui.split;
  $('morph-close').onclick=()=>$('morph-dialog').close();
  $('morph-kind').onchange=()=>ui.kind=$('morph-kind').value;
  $('morph-sources').onchange=()=>{ui.sources=$('morph-sources').value;morphSelection();};
  $('morph-split').onchange=()=>{ui.split=$('morph-split').value;morphSelection();};
  $('morph-count').onchange=()=>ui.count=Number($('morph-count').value);
  $('morph-seed').onchange=()=>ui.seed=Number($('morph-seed').value);
  $('morph-ratio-min').onchange=()=>ui.ratioMin=Number($('morph-ratio-min').value);
  $('morph-ratio-max').onchange=()=>ui.ratioMax=Number($('morph-ratio-max').value);
  for(const [id,key] of morphLimitFields)$(id).onchange=()=>ui.limits[key]=Number($(id).value);
  $('morph-hide').onchange=()=>{ui.hideRejected=$('morph-hide').checked;morphResults();};
  $('morph-generate').onclick=()=>morphAction(morphGenerate);
  $('morph-more').onclick=()=>{ui.seed+=1;$('morph-seed').value=ui.seed;morphAction(morphGenerate);};
  $('morph-select-all').onclick=()=>{morphVisible().forEach(index=>{if(!ui.added.has(index))ui.selected.add(index);});morphSelection();};
  $('morph-select-none').onclick=()=>{ui.selected.clear();morphSelection();};
  $('morph-deselect-rejected').onclick=()=>{[...ui.selected].forEach(index=>{if(!ui.candidates[index].filter.accepted)ui.selected.delete(index);});morphSelection();};
  $('morph-add').onclick=()=>morphAction(morphAddSelected);
  morphMessage(ui.message||'Выберите тип и запустите генерацию.');morphResults();morphControls();
}
async function morphAction(action) {
  if(morphUI.busy)return;
  morphUI.busy=true;morphControls();
  try {await action();} catch(error) {morphMessage(error.message,true);}
  finally {morphUI.busy=false;morphControls();}
}
function morphControls() {
  if(!$('morph-generate'))return;
  const ui=morphUI;
  document.querySelectorAll('#morph-dialog input, #morph-dialog select, #morph-dialog button').forEach(node=>node.disabled=ui.busy);
  $('morph-close').disabled=false;
  morphSelection();
}
function morphVisible() {
  return morphUI.candidates.map((_,index)=>index).filter(index=>!morphUI.hideRejected||morphUI.candidates[index].filter.accepted);
}
function morphDraw(canvas,matrix) {
  const ctx=canvas.getContext('2d'),size=morphCell*morphZoom;
  ctx.fillStyle='#000';ctx.fillRect(0,0,size,size);
  if(!matrix.length)return;
  const height=Math.min(matrix.length,morphCell),width=Math.min(matrix[0].length,morphCell);
  const x0=Math.floor((morphCell-width)/2),y0=Math.floor((morphCell-height)/2);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
    const gray=Math.round(255*Math.max(0,Math.min(1,matrix[y][x])));
    ctx.fillStyle=`rgb(${gray},${gray},${gray})`;ctx.fillRect((x0+x)*morphZoom,(y0+y)*morphZoom,morphZoom,morphZoom);
  }
}
function morphImperfections(values) {
  return `шум ${values.noise.toFixed(2)} · центр ${values.hotspot.toFixed(2)} · полосы ${values.stripes.toFixed(2)} · выбито ${values.dropout.toFixed(2)}`;
}
function morphResults() {
  const ui=morphUI,visible=morphVisible();
  $('morph-results').innerHTML=visible.length?visible.map(index=>{
    const candidate=ui.candidates[index],ok=candidate.filter.accepted;
    return `<article id="morph-card-${index}" class="card autogen-candidate morph-candidate">
      <label><input id="morph-choose-${index}" type="checkbox"> Объект ${index+1}</label>
      <canvas id="morph-preview-${index}" width="${morphCell*morphZoom}" height="${morphCell*morphZoom}" aria-label="Сгенерированный объект ${index+1}"></canvas>
      <p class="morph-verdict ${ok?'ok':'no'}">${esc(morphReason(candidate.filter))}</p>
      <p class="muted">из ${esc(candidate.source.split+' / '+candidate.source.id)} · рамка ${candidate.box_index+1} (${esc(morphKinds[candidate.source_kind]||candidate.source_kind)})</p>
      <p class="muted">${candidate.length} × ${candidate.width} (строк × датчиков) · ${morphImperfections(candidate.morph.imperfections)}</p><p id="morph-added-${index}" class="muted"></p></article>`;
  }).join(''):`<p class="empty">${ui.candidates.length?'Все кандидаты скрыты фильтром.':'После генерации здесь появятся объекты в одном масштабе (1 пиксель кадра = 8 точек).'}</p>`;
  visible.forEach(index=>{
    morphDraw($(`morph-preview-${index}`),ui.candidates[index].matrix);
    $(`morph-choose-${index}`).onchange=()=>{if($(`morph-choose-${index}`).checked)ui.selected.add(index);else ui.selected.delete(index);morphSelection();};
  });
  if(ui.library)$('morph-library').textContent=`Целых реальных объектов в ${ui.librarySplit} по типам (без обрубков): `+Object.entries(ui.library).map(([kind,count])=>`${morphKinds[kind]} ${count}`).join(', ')+'.';
  morphSelection();
}
function morphSelection() {
  const ui=morphUI;if(!$('morph-add'))return;
  const accepted=ui.candidates.filter(candidate=>candidate.filter.accepted).length;
  $('morph-count-info').textContent=ui.candidates.length?`Выбрано ${ui.selected.size} · фильтр одобрил ${accepted} / ${ui.candidates.length}`:'';
  $('morph-add').disabled=ui.busy||!ui.selected.size;
  $('morph-select-all').disabled=$('morph-select-none').disabled=ui.busy||!ui.candidates.length;
  $('morph-deselect-rejected').disabled=ui.busy||![...ui.selected].some(index=>!ui.candidates[index].filter.accepted);
  $('morph-generate').disabled=$('morph-more').disabled=ui.busy||(ui.sources==='bank'&&!morphBankSources().length);
  for(const index of morphVisible()) {
    $(`morph-choose-${index}`).checked=ui.selected.has(index);
    $(`morph-choose-${index}`).disabled=ui.busy||ui.added.has(index);
    $(`morph-card-${index}`).classList.toggle('chosen',ui.selected.has(index));
    $(`morph-added-${index}`).textContent=ui.added.has(index)?'В наборе':'';
  }
}
async function morphGenerate() {
  const ui=morphUI;
  const body={kind:ui.kind,count:ui.count,seed:ui.seed,ratio_min:ui.ratioMin,ratio_max:ui.ratioMax,limits:ui.limits,split:ui.split};
  if(ui.sources==='bank') {
    body.sources=morphBankSources();
    if(!body.sources.length)throw Error(`В наборе редактора нет целых реальных объектов из ${ui.split}: добавьте рамки кликом по исходному кадру этой части.`);
  }
  morphMessage('Генерация… Первый запуск собирает библиотеку реальных объектов, это займёт пару секунд.');
  const result=await synPost('morph-generate',body);
  ui.candidates=result.candidates;ui.library=result.library;ui.librarySplit=result.split;ui.selected.clear();ui.added.clear();
  morphResults();
  const accepted=result.candidates.filter(candidate=>candidate.filter.accepted).length;
  const short=result.candidates.length<result.requested?` Запрошено ${result.requested}: из этих исходников больше подходящих по форме не получилось.`:'';
  const notes=result.notes?.length?'\n'+result.notes.join('\n'):'';
  morphMessage(`Создано ${result.candidates.length} объектов «${morphKinds[ui.kind]}» из ${result.split}; фильтр одобрил ${accepted}.${short}${morphSifting(result.size_filter)} Отметьте понравившиеся.${notes}`);
}
async function morphAddSelected() {
  const ui=morphUI,indices=[...ui.selected].sort((a,b)=>a-b),added=[];
  for(const index of indices) {
    const candidate=ui.candidates[index];
    const object={source:synCopy(candidate.source),box_index:candidate.box_index,morph:synCopy(candidate.morph),
      transform:synDefaultTransform(),prepared:null,error:null,prepareToken:0};
    syntheticUI.objectBank.push(object);added.push(object);
    ui.added.add(index);ui.selected.delete(index);
  }
  syntheticUI.activeObject=syntheticUI.objectBank.length-1;
  synSyncTransformFields();synDraw();synButtons();
  morphMessage(`Добавляется ${added.length} объектов в набор…`);
  await Promise.all(added.map(object=>synPrepare(object)));
  const failed=added.filter(object=>object.error).length;
  $('morph-added-info').textContent=`В наборе редактора ${syntheticUI.objectBank.length} объектов.`;
  morphMessage(`Добавлено ${added.length-failed} объектов в набор редактора${failed?`; ${failed} с ошибкой`:''}. Их можно размещать вручную или отдать автогенератору.`,!!failed);
  morphSelection();
}
$('morph-open').onclick=openMorphGenerator;
