/* Page shell: the helpers editor.js and autogen.js share, the theme switch and start-up. */
'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// The editor scripts check which page section is open; this page has only the editor.
const section = 'Synthetic Editor';
function setSection() { renderSyntheticEditor(); }
async function request(url, options) {
  let response;
  // fetch rejects only when no answer arrived at all; the browser's own text for that is "Failed to fetch".
  try {response=await fetch(url,options);}
  catch(error) {throw Error('The editor server is not answering. Check the console where python -m ir_line_synth was started: if it has stopped, start it again and reload this page.');}
  const data=await response.json();
  if (!response.ok) throw Error(data.error||`HTTP ${response.status}`);
  return data;
}
$('theme').onclick=()=>{
  const current=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  const next=current==='dark'?'light':'dark';
  document.documentElement.dataset.theme=next;
  try {localStorage.setItem('ir-line-synth-theme',next);} catch(error) {}
};
// Deferred scripts have all run by now, so the editor renderer exists.
document.addEventListener('DOMContentLoaded',()=>renderSyntheticEditor());
